export function highlightText(text, phrases) {
  if (!phrases || phrases.length === 0) return text;

  let highlightedText = text;

  phrases.forEach((phrase) => {
    const regex = new RegExp(`(${phrase})`, "gi");
    highlightedText = highlightedText.replace(
      regex,
      `<mark class="phish-highlight">$1</mark>`
    );
  });

  return highlightedText;
}
