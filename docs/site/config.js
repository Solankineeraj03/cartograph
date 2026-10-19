// Set once: the public GitHub repository that hosts this site. Evidence links on the project page point into it.
window.CARTOGRAPH_REPO = "https://github.com/Solankineeraj03/cartograph";
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("a[data-repo]").forEach((a) => {
    const path = a.dataset.repo;
    a.href = path === "" ? window.CARTOGRAPH_REPO
      : `${window.CARTOGRAPH_REPO}/${path.endsWith("/") ? "tree" : "blob"}/main/${path}`;
    a.target = "_blank"; a.rel = "noopener";
  });
});
