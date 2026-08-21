import { REPORT_CACHE_STORAGE_KEY, SAMPLE_RESULTS } from "./appConfig";

export function getReportId(report) {
  return report?.sample_id || report?.run_id || null;
}

export function getReportPath(report) {
  const reportId = getReportId(report);
  return reportId ? `/analysis/${encodeURIComponent(reportId)}` : "/upload";
}

export function getSampleById(sampleId) {
  return SAMPLE_RESULTS.find((sample) => sample.id === sampleId) || null;
}

export function readCachedReports() {
  try {
    return JSON.parse(globalThis.window?.sessionStorage?.getItem(REPORT_CACHE_STORAGE_KEY) || "{}");
  } catch {
    return {};
  }
}

export function readCachedReport(reportId) {
  return readCachedReports()[reportId] || null;
}

export function cacheReport(report) {
  const reportId = getReportId(report);
  if (!reportId) {
    return;
  }

  const reports = readCachedReports();
  reports[reportId] = report;
  try {
    globalThis.window?.sessionStorage?.setItem(REPORT_CACHE_STORAGE_KEY, JSON.stringify(reports));
  } catch {
    // If session storage is blocked, direct sample routes and saved-analysis routes can still restore from their source.
  }
}
