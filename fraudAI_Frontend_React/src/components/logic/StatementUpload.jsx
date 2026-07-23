"use client"

import { useEffect, useState } from "react"
import axios from "axios"
import { auth } from "./firebase.js"
import Header from "./Header.jsx"
import SidebarContent from "./SidebarContent"
import { handleGoogleSignIn } from "./auth"
import { Upload, AlertTriangle, Loader2 } from "lucide-react"
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip } from "recharts"
import { API_BASE_URL } from "@/lib/api"

const SEVERITY_COLORS = {
  low: "#60a5fa",
  medium: "#f59e0b",
  high: "#ef4444",
}

const FlaggedDot = (props) => {
  const { cx, cy, payload, index } = props
  if (!payload?.flagged) return <circle key={`dot-${index}`} cx={cx} cy={cy} r={0} fill="transparent" />
  return <circle key={`dot-${index}`} cx={cx} cy={cy} r={4} fill="#ef4444" stroke="none" />
}

const StatementUpload = () => {
  const [user, setUser] = useState(null)
  const [file, setFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState("")
  const [report, setReport] = useState(null)

  useEffect(() => {
    const currentUser = auth.currentUser
    if (currentUser) setUser(currentUser)
  }, [])

  const handleFileChange = (e) => {
    setFile(e.target.files?.[0] || null)
    setError("")
    setReport(null)
  }

  const handleUpload = async () => {
    if (!file) return
    setUploading(true)
    setError("")
    setReport(null)
    try {
      const formData = new FormData()
      formData.append("file", file)
      const response = await axios.post(`${API_BASE_URL}/statement/analyze`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 90000,
      })
      setReport(response.data)
    } catch (err) {
      setError(err.response?.data?.error || "Failed to analyze statement. Please try again.")
    } finally {
      setUploading(false)
    }
  }

  const chartData = (report?.transactions || []).map((t) => ({
    date: new Date(t.date).toLocaleDateString(),
    amount: t.direction === "debit" ? -t.amount : t.amount,
    flagged: t.flagged,
  }))

  return (
    <div className="flex min-h-screen" style={{ background: "var(--background)", color: "var(--foreground)" }}>
      <aside className="hidden md:flex flex-col w-60 min-h-screen border-r border-[var(--border)]">
        <SidebarContent />
      </aside>

      <div className="flex-1 flex flex-col overflow-y-auto">
        <Header user={user} onSignIn={handleGoogleSignIn} />

        <div className="flex-1 p-8">
          <div className="mb-8">
            <div className="font-mono text-[10px] uppercase tracking-widest opacity-40 mb-1">Statement Analysis</div>
            <h1 className="font-display italic text-3xl font-black">Upload Bank Statement</h1>
            <p className="font-mono text-[11px] opacity-50 mt-2 max-w-2xl">
              Upload a CSV bank statement to surface fraud-relevant patterns and statistics computed
              fresh from your own transaction history. Nothing here is compared against other users'
              data, and no data is retained after analysis.
            </p>
          </div>

          {/* Upload box */}
          <div className="border border-[var(--border)] p-6 mb-8">
            <div className="flex items-center gap-4 flex-wrap">
              <label className="flex items-center gap-2 px-4 py-2 border border-[var(--border)] font-mono text-[11px] uppercase tracking-widest cursor-pointer hover:border-[var(--accent)] transition-colors">
                <Upload className="h-3.5 w-3.5" />
                Choose File
                <input type="file" accept=".csv" className="hidden" onChange={handleFileChange} />
              </label>
              <span className="font-mono text-[11px] opacity-60">
                {file ? file.name : "No file selected (.csv)"}
              </span>
              <button
                onClick={handleUpload}
                disabled={!file || uploading}
                className="ml-auto px-4 py-2 font-mono text-[10px] uppercase tracking-widest transition-all disabled:opacity-30"
                style={{ background: "var(--accent)", color: "#fff" }}
              >
                {uploading ? (
                  <span className="flex items-center gap-2">
                    <Loader2 className="h-3.5 w-3.5 animate-spin" /> Analyzing...
                  </span>
                ) : (
                  "Analyze Statement"
                )}
              </button>
            </div>
            {error && (
              <div className="mt-4 flex items-center gap-2 font-mono text-[11px]" style={{ color: "#ef4444" }}>
                <AlertTriangle className="h-3.5 w-3.5" /> {error}
              </div>
            )}
          </div>

          {report && (
            <>
              {/* Stat cards */}
              <div
                className="grid gap-px md:grid-cols-2 lg:grid-cols-4 border border-[var(--border)] mb-8"
                style={{ background: "var(--border)" }}
              >
                {[
                  { tag: "01", title: "Transactions", value: report.data_quality.parsed_transaction_count },
                  { tag: "02", title: "Total In", value: `+${report.summary.total_credit.toFixed(2)}` },
                  { tag: "03", title: "Total Out", value: `-${report.summary.total_debit.toFixed(2)}` },
                  { tag: "04", title: "Patterns Found", value: report.pattern_findings.length },
                ].map((item) => (
                  <div key={item.tag} className="p-6" style={{ background: "var(--background)" }}>
                    <div className="font-mono text-[10px] opacity-40 mb-4 uppercase tracking-widest">
                      [{item.tag}] {item.title}
                    </div>
                    <div className="text-3xl font-mono tracking-tight" style={{ color: "var(--accent)" }}>
                      {item.value}
                    </div>
                  </div>
                ))}
              </div>

              {/* Data quality note */}
              <div className="border border-[var(--border)] p-6 mb-8 font-mono text-[11px] opacity-60">
                <div className="uppercase tracking-widest opacity-40 mb-2 text-[10px]">Data Quality</div>
                Parsed {report.data_quality.parsed_transaction_count} of{" "}
                {report.data_quality.parsed_transaction_count + report.data_quality.rows_dropped_during_parsing} rows
                &middot; {report.data_quality.exact_duplicate_rows} exact duplicate row(s) &middot; anomaly scoring
                confidence: {report.anomaly_scoring_confidence}
              </div>

              {/* Chart */}
              <div className="border border-[var(--border)] p-6 mb-8">
                <div className="font-mono text-[10px] uppercase tracking-widest opacity-40 mb-6">
                  Transaction Amounts Over Time (flagged in red)
                </div>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={chartData}>
                    <XAxis dataKey="date" stroke="#888888" fontSize={10} tickLine={false} axisLine={false} minTickGap={30} />
                    <YAxis stroke="#888888" fontSize={12} tickLine={false} axisLine={false} />
                    <Tooltip
                      content={({ active, payload }) =>
                        active && payload?.length ? (
                          <div className="border border-[var(--border)] p-2" style={{ background: "var(--background)" }}>
                            <p
                              className="font-mono text-[11px]"
                              style={{ color: payload[0].payload.flagged ? "#ef4444" : "var(--accent)" }}
                            >
                              {payload[0].value.toFixed(2)} {payload[0].payload.flagged ? "(flagged)" : ""}
                            </p>
                          </div>
                        ) : null
                      }
                    />
                    <Line type="monotone" dataKey="amount" stroke="#3b82f6" strokeWidth={1.5} dot={<FlaggedDot />} />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Pattern findings */}
              <div className="border border-[var(--border)] mb-8">
                <div className="px-6 py-4 border-b border-[var(--border)] font-mono text-[10px] uppercase tracking-widest opacity-40">
                  Pattern Findings
                </div>
                {report.pattern_findings.length === 0 ? (
                  <div className="px-6 py-8 text-center font-mono text-[11px] uppercase tracking-widest opacity-30">
                    No notable patterns detected
                  </div>
                ) : (
                  <div className="divide-y divide-[var(--border)]">
                    {report.pattern_findings.map((finding, i) => (
                      <div key={i} className="px-6 py-4 flex items-start gap-3">
                        <span
                          className="mt-1 px-2 py-0.5 font-mono text-[9px] uppercase tracking-widest shrink-0"
                          style={{
                            background: `${SEVERITY_COLORS[finding.severity]}22`,
                            color: SEVERITY_COLORS[finding.severity],
                          }}
                        >
                          {finding.severity}
                        </span>
                        <div>
                          <div className="font-mono text-[10px] uppercase tracking-widest opacity-40 mb-1">
                            {finding.pattern_type.replaceAll("_", " ")}
                          </div>
                          <div className="font-mono text-xs opacity-80">{finding.description}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Flagged transactions table */}
              <div className="border border-[var(--border)] divide-y divide-[var(--border)]">
                <div className="grid grid-cols-12 px-6 py-3 font-mono text-[10px] uppercase tracking-widest opacity-40">
                  <span className="col-span-1">#</span>
                  <span className="col-span-4">Description</span>
                  <span className="col-span-3">Date</span>
                  <span className="col-span-2">Anomaly Score</span>
                  <span className="col-span-2 text-right">Amount</span>
                </div>
                {report.flagged_transactions.map((t, i) => (
                  <div
                    key={t.row_index}
                    className="grid grid-cols-12 px-6 py-4 items-center hover:bg-white/[0.02] transition-colors"
                  >
                    <span className="col-span-1 font-mono text-[10px] opacity-30">{String(i + 1).padStart(2, "0")}</span>
                    <span className="col-span-4 font-mono text-xs truncate">{t.description}</span>
                    <span className="col-span-3 font-mono text-[11px] opacity-50">
                      {new Date(t.date).toLocaleDateString()}
                    </span>
                    <span className="col-span-2 font-mono text-[11px]" style={{ color: "#ef4444" }}>
                      {(t.anomaly_score * 100).toFixed(0)}%
                    </span>
                    <span
                      className="col-span-2 text-right font-mono text-sm font-medium"
                      style={{ color: t.direction === "credit" ? "var(--accent)" : "#ef4444" }}
                    >
                      {t.direction === "credit" ? "+" : "-"}
                      {t.amount.toFixed(2)}
                    </span>
                  </div>
                ))}
                {report.flagged_transactions.length === 0 && (
                  <div className="px-6 py-12 text-center font-mono text-[11px] uppercase tracking-widest opacity-30">
                    No transactions flagged
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}

export default StatementUpload
