// Shared script for the Planform website pages: theme toggle and calculator helpers.
// Load in <head> so the saved theme applies before the page paints.

(function () {
  let saved = null;
  try { saved = localStorage.getItem("theme"); } catch (e) {}
  if (saved === "light" || saved === "dark") document.documentElement.dataset.theme = saved;
  document.addEventListener("DOMContentLoaded", () => {
    const btn = document.querySelector(".theme-btn");
    if (!btn) return;
    btn.addEventListener("click", () => {
      const cur = document.documentElement.dataset.theme ||
        (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
      const next = cur === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      try { localStorage.setItem("theme", next); } catch (e) {}
    });
  });
})();

const $ = id => document.getElementById(id);
const num = id => parseFloat($(id).value);
const fmt = (x, d = 0) => !isFinite(x) ? "—" : x.toLocaleString("en-US", {maximumFractionDigits: d, minimumFractionDigits: d});
function card(k, v, state, s) {
  return `<div class="r${state ? " " + state : ""}"><div class="k">${k}</div><div class="v">${v}</div>${s ? `<div class="s">${s}</div>` : ""}</div>`;
}
const safety = (n, lo = 1.5) => n >= lo ? "ok" : n >= 1 ? "warn" : "bad";
function bind(ids, fn) { ids.forEach(id => $(id).addEventListener("input", fn)); fn(); }
