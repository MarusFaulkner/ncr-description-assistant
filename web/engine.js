/*
 * Browser port of ncr_assistant (rules.py + keyword classifier).
 *
 * The vocabulary and regex patterns are NOT duplicated here: build_static.py
 * exports them from the Python modules into CFG, so the Python code stays the
 * single source of truth. Only the scoring logic is mirrored, and
 * tests/test_web_parity.py proves it matches Python on every dataset row.
 */
function makeEngine(CFG) {
  const esc = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  // Prefix match on a word boundary, same as Python r"\b" + re.escape(term).
  const findTerms = (text, terms) =>
    terms.filter((t) => new RegExp("\\b" + esc(t)).test(text));

  function grade(score) {
    if (score >= 75) return "Actionable";
    if (score >= 50) return "Needs detail";
    return "Rewrite";
  }

  function checkDescription(text) {
    const clean = (text || "").trim();
    const lower = clean.toLowerCase();
    const words = clean.match(/\S+/g) || [];

    const conditions = findTerms(lower, CFG.condition_terms);
    const hasLocation =
      findTerms(lower, CFG.location_terms).length > 0 ||
      CFG.location_patterns.some((p) => new RegExp(p).test(lower));
    const hasMeasurement = new RegExp(CFG.measurement_pattern, "i").test(lower);
    const vague = findTerms(lower, CFG.vague_terms);
    const cause = [];
    for (const p of CFG.cause_patterns) {
      for (const m of lower.matchAll(new RegExp(p, "g"))) cause.push(m[0]);
    }

    let score = 0;
    const feedback = [];

    if (conditions.length) score += 30;
    else feedback.push("Name the observable condition (e.g., leak, crack, missing, loose, out of tolerance).");

    if (hasLocation) score += 25;
    else feedback.push("Say where: the feature, side, phase, or position number.");

    if (hasMeasurement) score += 25;
    else feedback.push("Add how much: a measured value with units, or a count (e.g., 2.5 mm, qty 3, 4 of 12 bolts).");

    if (words.length >= CFG.min_words && words.length <= CFG.max_words) score += 20;
    else if (words.length < CFG.min_words)
      feedback.push(`Too short to act on (${words.length} words). Aim for ${CFG.min_words}-${CFG.max_words}.`);
    else
      feedback.push(`Long (${words.length} words). Keep the description to the observation; put history and context in the body of the report.`);

    if (vague.length && !conditions.length) {
      score -= 20;
      feedback.push(`Vague wording (${vague.join(", ")}) does not tell the reader what was seen. Replace it with the actual condition.`);
    }
    if (cause.length) {
      score -= 25;
      feedback.push(`Speculates about cause (${cause.join(", ")}). Record only what you observed; root cause belongs to the investigation.`);
    }

    score = Math.max(0, Math.min(100, score));
    if (!feedback.length) feedback.push("Clear, specific, observation-only. Good to submit.");

    return {
      text: clean, score, grade: grade(score),
      has_condition: conditions.length > 0, has_location: hasLocation,
      has_measurement: hasMeasurement, vague_terms: vague,
      cause_speculation: cause, feedback,
    };
  }

  function keywordClassify(text, topK = 3) {
    const lower = (text || "").toLowerCase();
    const hits = Object.entries(CFG.categories).map(([cat, kws]) => [cat, findTerms(lower, kws).length]);
    const total = hits.reduce((s, [, n]) => s + n, 0);
    if (total === 0) return [];
    // Stable sort keeps declaration order on ties, matching Python's sorted().
    return hits
      .sort((a, b) => b[1] - a[1])
      .slice(0, topK)
      .filter(([, n]) => n > 0)
      .map(([cat, n]) => [cat, n / total]);
  }

  return { checkDescription, keywordClassify };
}

if (typeof module !== "undefined") module.exports = { makeEngine };
