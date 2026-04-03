export const STATUSES = [
  { value: "skilt_funnet", label: "Skilt funnet" },
  { value: "uklart", label: "Uklart" },
  { value: "trenger_manuell", label: "Trenger manuell vurdering" },
] as const;

/** Trenings-/annotasjonslabel (API mapper til ReviewStatus). */
export const ANNOTATION_LABELS = [
  { value: "alarm_sign", label: "Alarm-skilt (alarm_sign)" },
  { value: "unclear", label: "Uklart (unclear)" },
  { value: "not_alarm_sign", label: "Ikke alarm-skilt (not_alarm_sign)" },
] as const;

export type AnnotationLabel = (typeof ANNOTATION_LABELS)[number]["value"];

export function reviewStatusForAnnotation(label: AnnotationLabel): string {
  if (label === "alarm_sign") return "skilt_funnet";
  if (label === "not_alarm_sign") return "trenger_manuell";
  return "uklart";
}

export function defaultAnnotationFromPredicted(predictedStatus: string): AnnotationLabel {
  if (predictedStatus === "skilt_funnet") return "alarm_sign";
  if (predictedStatus === "trenger_manuell") return "not_alarm_sign";
  return "unclear";
}

export const ERROR_TYPES = [
  { value: "feil_objekt", label: "Feil objekt" },
  { value: "darlig_vinkel", label: "Dårlig vinkel" },
  { value: "for_langt_unna", label: "For langt unna" },
  { value: "refleks_skygge", label: "Refleks/skygge" },
  { value: "lignende_fasadedetalj", label: "Lignende fasadedetalj" },
  { value: "skilt_delvis_skjult", label: "Skilt delvis skjult" },
  { value: "inngang_ikke_synlig", label: "Inngang ikke synlig" },
  { value: "bildekvalitet_for_darlig", label: "Bildekvalitet for dårlig" },
  { value: "annet", label: "Annet" },
];

export const LIBRARY_CATEGORIES = [
  { value: "positive", label: "Positive eksempler" },
  { value: "negative_irrelevant", label: "Negative / irrelevante objekter" },
  { value: "vanskelig", label: "Vanskelige tilfeller" },
  { value: "vinkel_variasjon", label: "Ulike vinkler" },
  { value: "lys_variasjon", label: "Ulike lysforhold" },
  { value: "delvis_skjult", label: "Delvis skjulte skilt" },
];
