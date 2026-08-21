const FEEDBACK_CATEGORIES = [
  {
    id: "strength",
    title: "Good",
    label: "Keep",
    description: "Things that are working well.",
    matchers: [
      "good",
      "strong arm",
      "clear jump",
      "confirms",
      "stays close",
      "stay close",
      "stays straight",
      "balanced",
      "mostly parallel",
      "visible upward arc",
      "fairly smooth",
    ],
  },
  {
    id: "focus",
    title: "Needs Work",
    label: "Fix",
    description: "The highest-value mechanics to clean up.",
    matchers: [
      "needs work",
      "limited",
      "too tilted",
      "looks flat",
      "drifts sideways",
      "is off",
      "not parallel",
      "not stable enough",
      "not perfectly aligned",
      "short",
      "uneven",
      "high;",
      "score was capped",
      "quality limits",
      "drift away",
      "staggered during",
      "not perfectly",
    ],
  },
  {
    id: "watch",
    title: "Watch",
    label: "Tune",
    description: "Acceptable or unclear items worth monitoring.",
    matchers: [
      "decent",
      "acceptable",
      "some ",
      "slightly",
      "could be",
      "a bit",
      "close, but",
      "has some",
      "could not be measured",
      "needs more",
      "not graded",
      "beta metric",
      "selected:",
    ],
  },
];

function classifyFeedbackItem(item) {
  const normalized = String(item).toLowerCase();
  const matchedCategory = FEEDBACK_CATEGORIES.find((category) =>
    category.matchers.some((matcher) => normalized.includes(matcher))
  );

  return matchedCategory || FEEDBACK_CATEGORIES[2];
}

export function groupFeedbackItems(feedback = []) {
  const grouped = FEEDBACK_CATEGORIES.map((category) => ({
    ...category,
    items: [],
  }));

  feedback.forEach((item) => {
    const category = classifyFeedbackItem(item);
    grouped.find((group) => group.id === category.id)?.items.push(item);
  });

  return grouped;
}
