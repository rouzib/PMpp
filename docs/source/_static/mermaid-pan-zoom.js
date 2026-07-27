(() => {
  "use strict";

  const enhanced = new WeakMap();
  const minimumScale = 0.2;
  const maximumScale = 12;
  let scanScheduled = false;

  const parseViewBox = (svg) => {
    const raw = svg.getAttribute("viewBox");
    if (raw) {
      const values = raw.trim().split(/[ ,]+/).map(Number);
      if (values.length === 4 && values.every(Number.isFinite)) return values;
    }

    const box = svg.getBBox();
    return [box.x, box.y, Math.max(box.width, 1), Math.max(box.height, 1)];
  };

  const writeViewBox = (svg, viewBox) => {
    svg.setAttribute("viewBox", viewBox.join(" "));
  };

  const ensureControls = (svg, actions) => {
    const host = svg.closest(".mermaid-container-fullscreen, .mermaid-container");
    if (!host) return;

    const existing = host.querySelector(".pmpp-mermaid-controls");
    if (existing?.pmppTargetSvg === svg) return;
    existing?.remove();

    const controls = document.createElement("div");
    controls.className = "pmpp-mermaid-controls";
    controls.setAttribute("role", "group");
    controls.setAttribute("aria-label", "Diagram pan and zoom controls");
    controls.pmppTargetSvg = svg;

    const addButton = (label, title, action) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = label;
      button.title = title;
      button.setAttribute("aria-label", title);
      button.addEventListener("click", (event) => {
        event.stopPropagation();
        action();
        svg.focus({ preventScroll: true });
      });
      controls.appendChild(button);
    };

    addButton("+", "Zoom in", () => actions.zoom(1.3));
    addButton("−", "Zoom out", () => actions.zoom(1 / 1.3));
    addButton("↺", "Reset diagram view", actions.reset);
    host.appendChild(controls);
  };

  const enhance = (svg) => {
    const existingActions = enhanced.get(svg);
    if (existingActions) {
      ensureControls(svg, existingActions);
      return;
    }

    let original;
    try {
      const saved = svg.dataset.pmppOriginalViewBox;
      original = saved ? saved.split(" ").map(Number) : parseViewBox(svg);
    } catch (_error) {
      return;
    }
    if (original.length !== 4 || !original.every(Number.isFinite)) return;

    svg.dataset.pmppOriginalViewBox = original.join(" ");
    svg.dataset.pmppPanZoom = "true";
    svg.setAttribute("tabindex", "0");
    svg.setAttribute(
      "aria-label",
      "Interactive diagram. Drag or use arrow keys to pan; use the wheel, plus, or minus to zoom; press zero to reset."
    );

    let viewBox = parseViewBox(svg);
    let drag = null;

    const reset = () => {
      viewBox = [...original];
      writeViewBox(svg, viewBox);
    };

    const zoom = (factor, clientX, clientY) => {
      const rect = svg.getBoundingClientRect();
      if (!rect.width || !rect.height) return;

      const currentScale = original[2] / viewBox[2];
      const nextScale = Math.min(maximumScale, Math.max(minimumScale, currentScale * factor));
      const appliedFactor = nextScale / currentScale;
      if (Math.abs(appliedFactor - 1) < 1e-6) return;

      const x = clientX ?? rect.left + rect.width / 2;
      const y = clientY ?? rect.top + rect.height / 2;
      const anchorX = viewBox[0] + ((x - rect.left) / rect.width) * viewBox[2];
      const anchorY = viewBox[1] + ((y - rect.top) / rect.height) * viewBox[3];
      const width = viewBox[2] / appliedFactor;
      const height = viewBox[3] / appliedFactor;

      viewBox = [
        anchorX - ((x - rect.left) / rect.width) * width,
        anchorY - ((y - rect.top) / rect.height) * height,
        width,
        height,
      ];
      writeViewBox(svg, viewBox);
    };

    const pan = (xFraction, yFraction) => {
      viewBox = [
        viewBox[0] + viewBox[2] * xFraction,
        viewBox[1] + viewBox[3] * yFraction,
        viewBox[2],
        viewBox[3],
      ];
      writeViewBox(svg, viewBox);
    };

    svg.addEventListener(
      "wheel",
      (event) => {
        event.preventDefault();
        zoom(Math.exp(-event.deltaY * 0.0015), event.clientX, event.clientY);
      },
      { passive: false }
    );

    svg.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) return;
      drag = { x: event.clientX, y: event.clientY, viewBox: [...viewBox] };
      svg.setPointerCapture(event.pointerId);
      svg.classList.add("is-panning");
    });

    svg.addEventListener("pointermove", (event) => {
      if (!drag) return;
      const rect = svg.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      viewBox = [
        drag.viewBox[0] - ((event.clientX - drag.x) / rect.width) * drag.viewBox[2],
        drag.viewBox[1] - ((event.clientY - drag.y) / rect.height) * drag.viewBox[3],
        drag.viewBox[2],
        drag.viewBox[3],
      ];
      writeViewBox(svg, viewBox);
    });

    const stopDragging = (event) => {
      if (!drag) return;
      drag = null;
      svg.classList.remove("is-panning");
      if (svg.hasPointerCapture(event.pointerId)) svg.releasePointerCapture(event.pointerId);
    };
    svg.addEventListener("pointerup", stopDragging);
    svg.addEventListener("pointercancel", stopDragging);

    svg.addEventListener("keydown", (event) => {
      const actions = {
        ArrowLeft: () => pan(-0.08, 0),
        ArrowRight: () => pan(0.08, 0),
        ArrowUp: () => pan(0, -0.08),
        ArrowDown: () => pan(0, 0.08),
        "+": () => zoom(1.3),
        "=": () => zoom(1.3),
        "-": () => zoom(1 / 1.3),
        "0": reset,
      };
      const action = actions[event.key];
      if (!action) return;
      event.preventDefault();
      action();
    });

    const actions = { reset, zoom };
    enhanced.set(svg, actions);
    ensureControls(svg, actions);
  };

  const scan = () => {
    scanScheduled = false;
    document.querySelectorAll(".mermaid svg").forEach(enhance);
  };

  const scheduleScan = () => {
    if (scanScheduled) return;
    scanScheduled = true;
    window.requestAnimationFrame(scan);
  };

  const start = () => {
    scheduleScan();
    const observer = new MutationObserver(scheduleScan);
    observer.observe(document.body, { childList: true, subtree: true });
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start, { once: true });
  } else {
    start();
  }
})();
