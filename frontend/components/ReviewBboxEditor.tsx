"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export type BboxNorm = { x: number; y: number; w: number; h: number };

type Props = {
  imageUrl: string;
  modelBbox: BboxNorm | null;
  value: BboxNorm | null;
  onChange: (b: BboxNorm | null) => void;
};

const ZOOM_MIN = 0.35;
const ZOOM_MAX = 8;
const VIEWPORT_H = 420;
const MIN_WORLD = 6;
/** Overlay tegnes i viewport-piksler (utenfor scale); hit-slop i verden = dette / zoom for jevn skjermflate. */
const HANDLE_SCREEN_PX = 12;
const BBOX_OVERLAY_BORDER_PX = 2;
const HIT_SLOP_SCREEN_PX = 18;

function hitSlopWorld(zoom: number): number {
  return HIT_SLOP_SCREEN_PX / Math.max(zoom, ZOOM_MIN);
}

function clamp(n: number, a: number, b: number) {
  return Math.max(a, Math.min(b, n));
}

type WorldBox = { l: number; t: number; r: number; b: number };
type HandleId = "nw" | "n" | "ne" | "e" | "se" | "s" | "sw" | "w";

function normToWorldBox(b: BboxNorm, dw: number, dh: number): WorldBox {
  return {
    l: b.x * dw,
    t: b.y * dh,
    r: (b.x + b.w) * dw,
    b: (b.y + b.h) * dh,
  };
}

function worldBoxToNorm(wb: WorldBox, dw: number, dh: number): BboxNorm {
  const l = clamp(Math.min(wb.l, wb.r), 0, dw);
  const t = clamp(Math.min(wb.t, wb.b), 0, dh);
  const r = clamp(Math.max(wb.l, wb.r), 0, dw);
  const btm = clamp(Math.max(wb.t, wb.b), 0, dh);
  const nw = Math.max(MIN_WORLD, r - l);
  const nh = Math.max(MIN_WORLD, btm - t);
  const l2 = clamp(l, 0, dw - nw);
  const t2 = clamp(t, 0, dh - nh);
  return { x: l2 / dw, y: t2 / dh, w: nw / dw, h: nh / dh };
}

function hitTest(ix: number, iy: number, wb: WorldBox, slop: number): HandleId | "inside" | null {
  const { l, t, r, b } = wb;
  const cx = (l + r) / 2;
  const cy = (t + b) / 2;
  const corners: [HandleId, number, number][] = [
    ["nw", l, t],
    ["ne", r, t],
    ["se", r, b],
    ["sw", l, b],
  ];
  for (const [h, hx, hy] of corners) {
    if (Math.abs(ix - hx) <= slop && Math.abs(iy - hy) <= slop) return h;
  }
  if (Math.abs(iy - t) <= slop && ix >= l - slop && ix <= r + slop) return "n";
  if (Math.abs(iy - b) <= slop && ix >= l - slop && ix <= r + slop) return "s";
  if (Math.abs(ix - l) <= slop && iy >= t - slop && iy <= b + slop) return "w";
  if (Math.abs(ix - r) <= slop && iy >= t - slop && iy <= b + slop) return "e";
  if (ix >= l && ix <= r && iy >= t && iy <= b) return "inside";
  return null;
}

function resizeByHandle(handle: HandleId, ix: number, iy: number, start: WorldBox): WorldBox {
  const { l, t, r, b } = start;
  switch (handle) {
    case "nw":
      return {
        l: Math.min(ix, r - MIN_WORLD),
        t: Math.min(iy, b - MIN_WORLD),
        r,
        b,
      };
    case "n":
      return { l, t: Math.min(iy, b - MIN_WORLD), r, b };
    case "ne":
      return { l, t: Math.min(iy, b - MIN_WORLD), r: Math.max(ix, l + MIN_WORLD), b };
    case "e":
      return { l, t, r: Math.max(ix, l + MIN_WORLD), b };
    case "se":
      return { l, t, r: Math.max(ix, l + MIN_WORLD), b: Math.max(iy, t + MIN_WORLD) };
    case "s":
      return { l, t, r, b: Math.max(iy, t + MIN_WORLD) };
    case "sw":
      return { l: Math.min(ix, r - MIN_WORLD), t, r, b: Math.max(iy, t + MIN_WORLD) };
    case "w":
      return { l: Math.min(ix, r - MIN_WORLD), t, r, b };
    default:
      return start;
  }
}

function cursorForHit(h: HandleId | "inside" | null): string {
  switch (h) {
    case "nw":
    case "se":
      return "nwse-resize";
    case "ne":
    case "sw":
      return "nesw-resize";
    case "n":
    case "s":
      return "ns-resize";
    case "e":
    case "w":
      return "ew-resize";
    case "inside":
      return "move";
    default:
      return "crosshair";
  }
}

type Interact =
  | { k: "new"; ax: number; ay: number; bx: number; by: number }
  | { k: "move"; i0: number; j0: number; bbox0: BboxNorm }
  | { k: "resize"; handle: HandleId; startWB: WorldBox };

export default function ReviewBboxEditor({ imageUrl, modelBbox, value, onChange }: Props) {
  const viewportRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const previewRef = useRef<HTMLCanvasElement>(null);

  const [natural, setNatural] = useState({ w: 1, h: 1 });
  const [drawn, setDrawn] = useState({ w: 320, h: 240 });
  const [view, setView] = useState({ zoom: 1, panX: 0, panY: 0 });
  const viewRef = useRef(view);
  viewRef.current = view;
  const { zoom, panX, panY } = view;

  const [interact, setInteract] = useState<Interact | null>(null);
  const [panDrag, setPanDrag] = useState<{ sx: number; sy: number; ox: number; oy: number } | null>(null);
  const [hoverHit, setHoverHit] = useState<HandleId | "inside" | null>(null);

  const effectiveBbox = value ?? modelBbox;

  const layoutImage = useCallback(() => {
    const vp = viewportRef.current;
    const img = imgRef.current;
    if (!vp || !img || !img.naturalWidth) return;
    const nw = img.naturalWidth;
    const nh = img.naturalHeight;
    const vr = vp.getBoundingClientRect();
    const maxW = Math.max(120, vr.width - 4);
    const maxH = Math.max(120, VIEWPORT_H - 4);
    const scale = Math.min(maxW / nw, maxH / nh, 1);
    const dw = nw * scale;
    const dh = nh * scale;
    setNatural({ w: nw, h: nh });
    setDrawn({ w: dw, h: dh });
  }, []);

  useEffect(() => {
    layoutImage();
    const ro = new ResizeObserver(() => layoutImage());
    if (viewportRef.current) ro.observe(viewportRef.current);
    window.addEventListener("resize", layoutImage);
    return () => {
      ro.disconnect();
      window.removeEventListener("resize", layoutImage);
    };
  }, [layoutImage, imageUrl]);

  useEffect(() => {
    setView({ zoom: 1, panX: 0, panY: 0 });
  }, [imageUrl]);

  useEffect(() => {
    setDrawn({ w: 320, h: 240 });
    setNatural({ w: 1, h: 1 });
  }, [imageUrl]);

  const viewportToWorld = (vx: number, vy: number) => {
    const { zoom: z, panX: px, panY: py } = viewRef.current;
    return { ix: (vx - px) / z, iy: (vy - py) / z };
  };

  const worldToNorm = useCallback(
    (ix0: number, iy0: number, ix1: number, iy1: number): BboxNorm => {
      const w = drawn.w;
      const h = drawn.h;
      const x0 = clamp(Math.min(ix0, ix1), 0, w);
      const y0 = clamp(Math.min(iy0, iy1), 0, h);
      const x1 = clamp(Math.max(ix0, ix1), 0, w);
      const y1 = clamp(Math.max(iy0, iy1), 0, h);
      return worldBoxToNorm({ l: x0, t: y0, r: x1, b: y1 }, w, h);
    },
    [drawn.h, drawn.w]
  );

  const normToWorldRect = useCallback(
    (b: BboxNorm) => ({
      left: b.x * drawn.w,
      top: b.y * drawn.h,
      width: b.w * drawn.w,
      height: b.h * drawn.h,
    }),
    [drawn.h, drawn.w]
  );

  const previewBbox: BboxNorm | null =
    interact?.k === "new"
      ? worldToNorm(interact.ax, interact.ay, interact.bx, interact.by)
      : effectiveBbox;

  const drawPreview = useCallback(() => {
    const canvas = previewRef.current;
    const img = imgRef.current;
    if (!canvas || !img || !img.complete || !natural.w) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const b = previewBbox;
    if (!b || b.w < 1e-4 || b.h < 1e-4) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }
    const sx = b.x * natural.w;
    const sy = b.y * natural.h;
    const sw = b.w * natural.w;
    const sh = b.h * natural.h;
    const cw = 200;
    const ch = 200;
    if (canvas.width !== cw) canvas.width = cw;
    if (canvas.height !== ch) canvas.height = ch;
    ctx.fillStyle = "var(--surface)";
    ctx.fillRect(0, 0, cw, ch);
    const ar = sw / Math.max(sh, 1e-6);
    let dw = cw;
    let dh = ch;
    if (ar > 1) dh = cw / ar;
    else dw = ch * ar;
    const ox = (cw - dw) / 2;
    const oy = (ch - dh) / 2;
    try {
      ctx.drawImage(img, sx, sy, sw, sh, ox, oy, dw, dh);
    } catch {
      ctx.clearRect(0, 0, cw, ch);
    }
    ctx.strokeStyle = "var(--accent)";
    ctx.lineWidth = 2;
    ctx.strokeRect(ox, oy, dw, dh);
  }, [natural.w, natural.h, previewBbox]);

  useEffect(() => {
    drawPreview();
  }, [drawPreview, imageUrl, interact, effectiveBbox]);

  const zoomAt = (vx: number, vy: number, factor: number) => {
    setView((v) => {
      const z0 = v.zoom;
      const z1 = clamp(z0 * factor, ZOOM_MIN, ZOOM_MAX);
      const ix = (vx - v.panX) / z0;
      const iy = (vy - v.panY) / z0;
      return { zoom: z1, panX: vx - ix * z1, panY: vy - iy * z1 };
    });
  };

  const onWheelViewport = (e: React.WheelEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const vp = viewportRef.current;
    if (!vp) return;
    const r = vp.getBoundingClientRect();
    zoomAt(e.clientX - r.left, e.clientY - r.top, e.deltaY > 0 ? 0.92 : 1.08);
  };

  const onMouseDownViewport = (e: React.MouseEvent) => {
    const vp = viewportRef.current;
    if (!vp) return;
    const r = vp.getBoundingClientRect();
    const vx = e.clientX - r.left;
    const vy = e.clientY - r.top;

    if (e.button === 1 || e.altKey || e.shiftKey) {
      e.preventDefault();
      const v = viewRef.current;
      setPanDrag({ sx: e.clientX, sy: e.clientY, ox: v.panX, oy: v.panY });
      return;
    }
    if (e.button !== 0) return;

    const { ix, iy } = viewportToWorld(vx, vy);
    const dw = drawn.w;
    const dh = drawn.h;
    const z = viewRef.current.zoom;
    const slop = hitSlopWorld(z);

    if (effectiveBbox) {
      const wb = normToWorldBox(effectiveBbox, dw, dh);
      const hit = hitTest(ix, iy, wb, slop);
      if (hit && hit !== "inside") {
        setInteract({ k: "resize", handle: hit as HandleId, startWB: { ...wb } });
        return;
      }
      if (hit === "inside") {
        setInteract({ k: "move", i0: ix, j0: iy, bbox0: { ...effectiveBbox } });
        return;
      }
    }

    setInteract({ k: "new", ax: ix, ay: iy, bx: ix, by: iy });
  };

  const onMouseMoveViewport = (e: React.MouseEvent) => {
    const vp = viewportRef.current;
    if (!vp || interact || panDrag) return;
    const r = vp.getBoundingClientRect();
    const { ix, iy } = viewportToWorld(e.clientX - r.left, e.clientY - r.top);
    const z = viewRef.current.zoom;
    const slop = hitSlopWorld(z);
    if (effectiveBbox) {
      const wb = normToWorldBox(effectiveBbox, drawn.w, drawn.h);
      setHoverHit(hitTest(ix, iy, wb, slop));
    } else {
      setHoverHit(null);
    }
  };

  useEffect(() => {
    function onMove(e: MouseEvent) {
      if (panDrag) {
        setView({
          ...viewRef.current,
          panX: panDrag.ox + (e.clientX - panDrag.sx),
          panY: panDrag.oy + (e.clientY - panDrag.sy),
        });
        return;
      }
      if (!interact || !viewportRef.current) return;
      const r = viewportRef.current.getBoundingClientRect();
      const { ix, iy } = viewportToWorld(e.clientX - r.left, e.clientY - r.top);
      const dw = drawn.w;
      const dh = drawn.h;

      if (interact.k === "new") {
        setInteract({ ...interact, bx: ix, by: iy });
        return;
      }
      if (interact.k === "move") {
        const dx = (ix - interact.i0) / dw;
        const dy = (iy - interact.j0) / dh;
        const b0 = interact.bbox0;
        const x = clamp(b0.x + dx, 0, 1 - b0.w);
        const y = clamp(b0.y + dy, 0, 1 - b0.h);
        onChange({ ...b0, x, y });
        return;
      }
      if (interact.k === "resize") {
        const wb2 = resizeByHandle(interact.handle, ix, iy, interact.startWB);
        const nb = worldBoxToNorm(wb2, dw, dh);
        onChange(nb);
      }
    }
    function onUp() {
      if (panDrag) {
        setPanDrag(null);
        return;
      }
      if (!interact) return;
      if (interact.k === "new") {
        const b = worldToNorm(interact.ax, interact.ay, interact.bx, interact.by);
        if (b.w > 0.002 && b.h > 0.002) onChange(b);
      }
      setInteract(null);
    }
    if (interact || panDrag) {
      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
      return () => {
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);
      };
    }
  }, [interact, panDrag, drawn.h, drawn.w, onChange, worldToNorm]);

  const rectWorld =
    interact?.k === "new"
      ? (() => {
          const l = Math.min(interact.ax, interact.bx);
          const t = Math.min(interact.ay, interact.by);
          return { left: l, top: t, width: Math.abs(interact.bx - interact.ax), height: Math.abs(interact.by - interact.ay) };
        })()
      : effectiveBbox
        ? normToWorldRect(effectiveBbox)
        : null;

  const renderHandles = rectWorld && effectiveBbox && interact?.k !== "new";
  const rectScreen =
    rectWorld != null
      ? {
          left: rectWorld.left * zoom + panX,
          top: rectWorld.top * zoom + panY,
          width: rectWorld.width * zoom,
          height: rectWorld.height * zoom,
        }
      : null;

  const vpCursor = panDrag ? "grabbing" : interact ? "crosshair" : cursorForHit(hoverHit);

  return (
    <div style={{ marginTop: 8 }}>
      <p className="muted" style={{ fontSize: 12 }}>
        <strong>Zoom:</strong> mushjul · <strong>Pan:</strong> midtklikk / Alt / Shift+dra ·{" "}
        <strong>Bbox:</strong> dra på tom flate · <strong>Flytt:</strong> dra inni boksen ·{" "}
        <strong>Skaler:</strong> hjørner og kanter — overlay følger zoom i bildekoordinater; håndtak og treffflate er i skjerm-piksler.
      </p>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 6 }}>
        <button type="button" className="secondary" onClick={() => onChange(modelBbox ? { ...modelBbox } : null)}>
          Bruk modellforslag
        </button>
        <button type="button" className="secondary" onClick={() => onChange(null)}>
          Fjern bbox
        </button>
        <button type="button" className="secondary" onClick={() => setView({ zoom: 1, panX: 0, panY: 0 })}>
          Reset zoom/pan
        </button>
      </div>
      <div style={{ display: "flex", gap: 16, flexWrap: "wrap", alignItems: "flex-start" }}>
        <div
          ref={viewportRef}
          style={{
            position: "relative",
            width: "100%",
            maxWidth: 720,
            height: VIEWPORT_H,
            overflow: "hidden",
            background: "var(--border)",
            borderRadius: 6,
            cursor: vpCursor,
            touchAction: "none",
          }}
          onWheel={onWheelViewport}
          onMouseDown={onMouseDownViewport}
          onMouseMove={onMouseMoveViewport}
          onMouseLeave={() => setHoverHit(null)}
          onContextMenu={(e) => e.preventDefault()}
        >
          <div
            style={{
              position: "absolute",
              left: 0,
              top: 0,
              width: drawn.w,
              height: drawn.h,
              transform: `translate(${panX}px, ${panY}px) scale(${zoom})`,
              transformOrigin: "0 0",
              pointerEvents: "none",
            }}
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              key={imageUrl}
              ref={imgRef}
              src={imageUrl}
              alt=""
              width={drawn.w}
              height={drawn.h}
              style={{
                width: drawn.w,
                height: drawn.h,
                display: "block",
                userSelect: "none",
                pointerEvents: "none",
              }}
              onLoad={layoutImage}
              draggable={false}
            />
          </div>
          {rectScreen && (
            <div
              style={{
                position: "absolute",
                left: 0,
                top: 0,
                right: 0,
                bottom: 0,
                pointerEvents: "none",
                zIndex: 1,
              }}
            >
              <div
                style={{
                  position: "absolute",
                  left: rectScreen.left,
                  top: rectScreen.top,
                  width: Math.max(0, rectScreen.width),
                  height: Math.max(0, rectScreen.height),
                  border:
                    interact?.k === "new"
                      ? `${BBOX_OVERLAY_BORDER_PX}px dashed var(--warn)`
                      : `${BBOX_OVERLAY_BORDER_PX}px solid var(--accent)`,
                  boxSizing: "border-box",
                  pointerEvents: "none",
                }}
              />
              {renderHandles &&
                rectWorld &&
                (["nw", "n", "ne", "e", "se", "s", "sw", "w"] as HandleId[]).map((hid) => {
                  const { left, top, width, height } = rectWorld;
                  const cx = left + width / 2;
                  const cy = top + height / 2;
                  const pos: Record<HandleId, [number, number]> = {
                    nw: [left, top],
                    n: [cx, top],
                    ne: [left + width, top],
                    e: [left + width, cy],
                    se: [left + width, top + height],
                    s: [cx, top + height],
                    sw: [left, top + height],
                    w: [left, cy],
                  };
                  const [wx, wy] = pos[hid];
                  const hl = HANDLE_SCREEN_PX;
                  const sl = wx * zoom + panX - hl / 2;
                  const st = wy * zoom + panY - hl / 2;
                  return (
                    <div
                      key={hid}
                      style={{
                        position: "absolute",
                        left: sl,
                        top: st,
                        width: hl,
                        height: hl,
                        background: "var(--accent)",
                        border: "1px solid var(--text)",
                        boxSizing: "border-box",
                        pointerEvents: "none",
                      }}
                    />
                  );
                })}
            </div>
          )}
        </div>
        <div className="card" style={{ padding: 10, minWidth: 220, maxWidth: 240 }}>
          <p className="muted" style={{ fontSize: 11, marginBottom: 8 }}>
            Forhåndsvisning av markert boks
          </p>
          {previewBbox && previewBbox.w > 0.002 && previewBbox.h > 0.002 ? (
            <canvas
              ref={previewRef}
              style={{ display: "block", width: 200, height: 200, borderRadius: 4, background: "var(--surface)" }}
            />
          ) : (
            <div
              className="muted"
              style={{
                width: 200,
                height: 200,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: 12,
                border: "1px dashed var(--border)",
                borderRadius: 4,
              }}
            >
              Ingen bbox
            </div>
          )}
        </div>
      </div>
      {value && (
        <p className="muted" style={{ fontSize: 11, marginTop: 4 }}>
          Manuell bbox: {JSON.stringify(value)}
        </p>
      )}
    </div>
  );
}
