import { Suspense } from "react";
import LibraryGallery from "./LibraryGallery";

export const dynamic = "force-dynamic";

export default function LibraryPage() {
  return (
    <Suspense fallback={<p className="muted">Laster bibliotek …</p>}>
      <LibraryGallery />
    </Suspense>
  );
}
