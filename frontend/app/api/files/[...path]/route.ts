import { NextRequest, NextResponse } from "next/server";

/** Samme oppløsning som lib/api.ts (API_BASE), ellers treffer proxien feil vert når .env mangler. */
function backendBase(): string {
  return (
    process.env.API_PROXY_TARGET ||
    process.env.NEXT_PUBLIC_API_URL ||
    "http://localhost:8000"
  ).replace(/\/$/, "");
}

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

function normalizePathSegments(path: string | string[] | undefined): string[] {
  if (path == null) return [];
  return Array.isArray(path) ? path : [path];
}

export async function GET(
  _request: NextRequest,
  context: { params: Promise<{ path: string | string[] }> },
) {
  const { path: rawPath } = await context.params;
  const segments = normalizePathSegments(rawPath);
  if (!segments.length) {
    return new NextResponse("Not found", { status: 404 });
  }
  const suffix = segments.map(encodeURIComponent).join("/");
  const url = `${backendBase()}/api/files/${suffix}`;
  let res: Response;
  try {
    res = await fetch(url, { cache: "no-store" });
  } catch {
    return new NextResponse("Upstream unreachable", { status: 502 });
  }
  if (!res.ok) {
    const text = await res.text();
    return new NextResponse(text, { status: res.status });
  }
  const buf = await res.arrayBuffer();
  const headers = new Headers();
  const ct = res.headers.get("content-type");
  if (ct) headers.set("Content-Type", ct);
  const cd = res.headers.get("content-disposition");
  if (cd) headers.set("Content-Disposition", cd);
  const cl = res.headers.get("content-length");
  if (cl) headers.set("Content-Length", cl);
  return new NextResponse(buf, { status: res.status, headers });
}
