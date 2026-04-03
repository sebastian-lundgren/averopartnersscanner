"use client";

import { useEffect, useState } from "react";
import { fileUrl } from "@/lib/api";
import type { BboxLike } from "@/lib/annotationLearning";

const OUT = 76;

type Props = {
  imageId: number;
  bbox: BboxLike | null;
  source: "manual" | "model" | "none";
};

export default function AnnotationCropPreview({ imageId, bbox, source }: Props) {
  const [dataUrl, setDataUrl] = useState<string | null>(null);
  const [err, setErr] = useState(false);

  useEffect(() => {
    if (!bbox) {
      setDataUrl(null);
      setErr(false);
      return;
    }
    setDataUrl(null);
    setErr(false);

    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      const nw = img.naturalWidth;
      const nh = img.naturalHeight;
      if (!nw || !nh) {
        setErr(true);
        return;
      }
      const sx = Math.max(0, bbox.x * nw);
      const sy = Math.max(0, bbox.y * nh);
      const sw = Math.max(1, Math.min(nw - sx, bbox.w * nw));
      const sh = Math.max(1, Math.min(nh - sy, bbox.h * nh));
      const c = document.createElement("canvas");
      c.width = OUT;
      c.height = OUT;
      const ctx = c.getContext("2d");
      if (!ctx) {
        setErr(true);
        return;
      }
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "high";
      ctx.drawImage(img, sx, sy, sw, sh, 0, 0, OUT, OUT);
      setDataUrl(c.toDataURL("image/jpeg", 0.85));
    };
    img.onerror = () => setErr(true);
    img.src = fileUrl(imageId, "original");
  }, [imageId, bbox?.x, bbox?.y, bbox?.w, bbox?.h]);

  if (!bbox || source === "none") {
    return (
      <div
        style={{
          width: OUT,
          height: OUT,
          background: "var(--muted-bg, #eee)",
          borderRadius: 4,
          border: "1px solid var(--border)",
          fontSize: 10,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          textAlign: "center",
          padding: 4,
          color: "var(--muted)",
        }}
      >
        Ingen bbox
      </div>
    );
  }

  if (err) {
    return (
      <div
        style={{
          width: OUT,
          height: OUT,
          fontSize: 10,
          display: "flex",
          alignItems: "center",
          color: "var(--muted)",
        }}
      >
        Kunne ikke laste bilde
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 2, alignItems: "flex-start" }}>
      {dataUrl ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={dataUrl}
          alt=""
          width={OUT}
          height={OUT}
          style={{
            width: OUT,
            height: OUT,
            objectFit: "cover",
            borderRadius: 4,
            border: "1px solid var(--border)",
            display: "block",
          }}
        />
      ) : (
        <div
          style={{
            width: OUT,
            height: OUT,
            background: "var(--muted-bg, #eee)",
            borderRadius: 4,
            border: "1px solid var(--border)",
          }}
        />
      )}
      <span className="muted" style={{ fontSize: 10, lineHeight: 1.2 }}>
        {source === "manual" ? "Manuell bbox" : "Modell (fallback)"}
      </span>
    </div>
  );
}
