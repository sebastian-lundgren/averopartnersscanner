"use client";

export default function LibraryDetailImage({ src }: { src: string }) {
  return (
    <img
      src={src}
      alt=""
      style={{ width: "100%", maxWidth: "100%", height: "auto", display: "block" }}
    />
  );
}
