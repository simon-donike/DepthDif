document.addEventListener("DOMContentLoaded", () => {
  const clearTabTitles = () => {
    // Material adds title attributes to nav tabs; removing them suppresses the
    // browser's default hover tooltip without changing the visible tab labels.
    document.querySelectorAll(".md-tabs__link[title]").forEach((tabLink) => {
      tabLink.removeAttribute("title");
    });
  };

  clearTabTitles();
  new MutationObserver(() => {
    clearTabTitles();
  }).observe(document.body, { childList: true, subtree: true });
});
