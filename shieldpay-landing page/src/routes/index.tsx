import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useState } from "react";
import dashboardPreview from "@/assets/dashboard-preview.jpg";
import { handleGoogleSignIn } from "@/components/logic/auth";
import { auth } from "@/components/logic/firebase";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      {
        property: "og:image",
        content:
          "https://id-preview--b5d92a96-f792-47a5-9914-6f5d392c71b4.lovable.app/og-image.jpg",
      },
    ],
  }),
  component: Index,
});

const PIPELINE = [
  {
    tag: "[INPUT]",
    title: "Raw Data Stream",
    body: "20+ behavioral signals ingested per transaction — device, geo, velocity, biometrics.",
  },
  {
    tag: "[PHASE 01]",
    title: "GAN Synthesis",
    body: "Generative adversarial networks simulate future fraud vectors to harden the classifier.",
  },
  {
    tag: "[PHASE 02]",
    title: "Random Forest",
    body: "High-confidence classification through distributed, weighted decision ensembles.",
  },
  {
    tag: "[OUTPUT]",
    title: "Binary Verdict",
    body: "Sub-40ms allow / challenge / block payload with full interpretability logs.",
  },
];

const SIGNALS = [
  { n: "01", label: "Transaction Amount Anomalies" },
  { n: "02", label: "Velocity & Frequency Drift" },
  { n: "03", label: "Recipient Verification State" },
  { n: "04", label: "Recipient Blacklist Match" },
  { n: "05", label: "Device Fingerprinting (Cross-Browser)" },
  { n: "06", label: "VPN / Proxy Detection" },
  { n: "07", label: "Geo-Location Impossibility" },
  { n: "08", label: "Behavioral Biometrics (Keystroke, Pointer)" },
  { n: "09", label: "Time-Since-Last with Recipient" },
  { n: "10", label: "Social Trust Score" },
  { n: "11", label: "Account Age & Provenance" },
  { n: "12", label: "High-Risk Transaction Window" },
  { n: "13", label: "Prior Fraud Complaint History" },
  { n: "14", label: "Location-Inconsistent Session" },
  { n: "15", label: "Normalized Peer-Cohort Amount" },
  { n: "16", label: "Spending Context Anomaly" },
  { n: "17", label: "Merchant Category Mismatch" },
  { n: "18", label: "Daily Limit Threshold" },
  { n: "19", label: "Recent High-Value Flag" },
  { n: "20", label: "Session Entropy Score" },
];

const METRICS = [
  { k: "Verdict Latency", v: "38ms", note: "P99 real-time" },
  { k: "Detection Recall", v: "99.98%", note: "GAN-augmented" },
  { k: "False Positive", v: "<0.01%", note: "Industry-low friction" },
  { k: "Protected Volume", v: "$4.2B", note: "Monthly throughput" },
];

const PRICING = [
  {
    tier: "Developer",
    price: "$0",
    per: "/ mo",
    note: "Up to 1,000 verdicts / month",
    features: ["Random Forest core", "Public REST API", "Community support"],
    cta: "Start building",
    featured: false,
  },
  {
    tier: "Growth",
    price: "$499",
    per: "/ mo",
    note: "Up to 250,000 verdicts / month",
    features: [
      "GAN-augmented training",
      "20+ behavioral signals",
      "Custom rule overlays",
      "SLA-backed uptime",
    ],
    cta: "Deploy Growth",
    featured: true,
  },
  {
    tier: "Enterprise",
    price: "Custom",
    per: "",
    note: "Unlimited high-volume decisioning",
    features: [
      "Dedicated model instance",
      "On-prem or VPC deploy",
      "24/7 risk desk",
      "Compliance & audit exports",
    ],
    cta: "Contact sales",
    featured: false,
  },
];

function Index() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);

  const onDemo = async () => {
    setLoading(true);
    try {
      await handleGoogleSignIn();
      if (auth.currentUser) navigate({ to: "/dashboard" });
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-background text-foreground font-sans selection:bg-accent selection:text-white">
      {/* NAV */}
      <nav className="sticky top-0 z-50 flex items-center justify-between px-6 md:px-10 py-4 border-b border-border bg-background/80 backdrop-blur-md">
        <div className="font-mono text-xs tracking-widest uppercase opacity-70">
          Shieldpay <span className="opacity-40">®</span> / AI.FRAUD
        </div>
        <div className="hidden md:flex gap-8 items-center">
          <a href="#pipeline" className="text-[10px] uppercase tracking-widest font-mono hover:text-accent transition-colors">Pipeline</a>
          <a href="#signals" className="text-[10px] uppercase tracking-widest font-mono hover:text-accent transition-colors">Signals</a>
          <a href="#dashboard" className="text-[10px] uppercase tracking-widest font-mono hover:text-accent transition-colors">Interface</a>
          <a href="#pricing" className="text-[10px] uppercase tracking-widest font-mono hover:text-accent transition-colors">Pricing</a>
          <button
            onClick={onDemo}
            disabled={loading}
            className="px-4 py-1.5 border border-foreground/20 text-[10px] uppercase tracking-widest font-mono hover:bg-foreground hover:text-background transition-all disabled:opacity-50"
          >
            {loading ? "Signing in.." : "Quick Demo"}
          </button>
        </div>
      </nav>

      {/* HERO */}
      <section className="relative px-6 md:px-10 pt-24 md:pt-32 pb-24 overflow-hidden border-b border-border">
        <div className="absolute inset-0 pointer-events-none opacity-20">
          <div className="absolute inset-0 bg-[linear-gradient(to_right,#ffffff0a_1px,transparent_1px),linear-gradient(to_bottom,#ffffff0a_1px,transparent_1px)] bg-[size:4rem_4rem]" />
        </div>
        <div className="absolute -top-40 -right-40 size-[600px] rounded-full bg-accent/10 blur-[120px] pointer-events-none" />

        <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-12 relative">
          <div className="lg:col-span-8 animate-reveal">
            <div className="inline-flex items-center gap-2 mb-8 font-mono text-[10px] uppercase tracking-widest opacity-60">
              <span className="relative flex h-2 w-2">
                <span className="absolute inline-flex h-full w-full rounded-full bg-accent opacity-75 animate-ping" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-accent" />
              </span>
              Live network / v4.2 production
            </div>
            <h1 className="font-display text-[min(14vw,140px)] font-black tracking-tighter leading-[0.85] italic mb-10 text-balance">
              SHIELD
              <br />
              <span className="text-accent">PAY.</span>
            </h1>
            <p className="max-w-[46ch] text-xl md:text-2xl text-muted font-display italic leading-relaxed text-pretty">
              Real-time adversarial defense powered by GAN-driven behavioral
              synthesis. We detect what others don't even see yet.
            </p>
            <div className="mt-10 flex flex-wrap gap-4">
              <button
                onClick={onDemo}
                disabled={loading}
                className="px-6 py-4 bg-accent text-white font-mono text-xs uppercase tracking-widest hover:brightness-110 transition-all disabled:opacity-50"
              >
                {loading ? "Signing in.." : "Quick Demo"}
              </button>
              <button className="px-6 py-4 border border-border font-mono text-xs uppercase tracking-widest hover:bg-white/5 transition-all">
                Read API Docs
              </button>
            </div>
          </div>

          <div className="lg:col-span-4 flex flex-col justify-end gap-4 animate-reveal [animation-delay:150ms]">
            <div className="p-6 border border-border bg-white/5 backdrop-blur-sm relative overflow-hidden">
              <div className="absolute top-0 left-0 w-full h-0.5 bg-accent/40 animate-scan" />
              <div className="flex justify-between items-end mb-8">
                <div className="font-mono text-[10px] uppercase opacity-40">Live Verdict</div>
                <div className="size-2 bg-accent shadow-[0_0_12px_var(--color-accent)]" />
              </div>
              <div className="text-5xl font-mono mb-2 tracking-tighter">0.038s</div>
              <div className="font-mono text-[10px] uppercase opacity-60 leading-tight">
                Avg latency · real-time analysis
              </div>
            </div>
            <div className="grid grid-cols-2 gap-px bg-border border border-border">
              <div className="bg-background p-4">
                <div className="font-mono text-[10px] opacity-40 mb-2">RECALL</div>
                <div className="font-mono text-lg tracking-tight">99.98%</div>
              </div>
              <div className="bg-background p-4">
                <div className="font-mono text-[10px] opacity-40 mb-2">SIGNALS</div>
                <div className="font-mono text-lg tracking-tight">20+</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* METRICS STRIP */}
      <section className="border-b border-border">
        <div className="max-w-7xl mx-auto grid grid-cols-2 md:grid-cols-4 divide-x divide-border">
          {METRICS.map((m, i) => (
            <div key={m.k} className="p-8">
              <p className="text-[10px] font-mono opacity-40 mb-3 uppercase tracking-widest">
                ({String(i + 1).padStart(2, "0")}) {m.k}
              </p>
              <p className="text-3xl font-mono font-medium tracking-tight">{m.v}</p>
              <p className="text-xs text-muted mt-2 font-mono">{m.note}</p>
            </div>
          ))}
        </div>
      </section>

      {/* PIPELINE */}
      <section id="pipeline" className="py-24 md:py-32 px-6 md:px-10 border-b border-border">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center gap-4 mb-16">
            <span className="font-mono text-xs opacity-40">01 / 04</span>
            <h2 className="text-sm uppercase tracking-[0.3em] font-bold">
              The Processing Pipeline
            </h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-px bg-border border border-border">
            {PIPELINE.map((p, i) => (
              <div
                key={p.tag}
                className={`bg-background p-8 min-h-[240px] flex flex-col justify-between ${
                  i === PIPELINE.length - 1 ? "bg-accent/[0.04]" : ""
                }`}
              >
                <div className="font-mono text-[10px] mb-8 text-accent">{p.tag}</div>
                <div>
                  <h3 className="text-2xl font-display italic mb-4">{p.title}</h3>
                  <p className="text-xs text-muted leading-relaxed font-mono">{p.body}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* DASHBOARD PREVIEW */}
      <section id="dashboard" className="py-24 md:py-32 px-6 md:px-10 bg-white/[0.02] border-b border-border">
        <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-16 lg:gap-24 items-center">
          <div>
            <span className="font-mono text-[10px] uppercase tracking-widest text-accent mb-6 block">
              02 / Interface
            </span>
            <h2 className="text-5xl md:text-6xl font-display font-black italic tracking-tighter mb-8">
              Signals of intention.
            </h2>
            <p className="text-muted mb-10 max-w-[48ch] leading-relaxed">
              Every verdict is inspectable. Trace which of the twenty signals
              drove the outcome — not a black box, a reasoned defense.
            </p>
            <div className="space-y-2">
              {[
                { n: "01", label: "Device Fingerprinting (Cross-Browser)", active: true },
                { n: "02", label: "Behavioral Biometrics (Keystroke Rhythm)" },
                { n: "03", label: "Velocity & Pattern Drift" },
                { n: "04", label: "Geo-Location Impossibility" },
                { n: "05", label: "Recipient Trust Graph" },
              ].map((s) => (
                <div
                  key={s.n}
                  className={`flex items-center gap-4 border-l-2 pl-6 py-3 transition-colors ${
                    s.active
                      ? "border-accent"
                      : "border-border opacity-50 hover:opacity-100 hover:border-foreground/40"
                  }`}
                >
                  <span className="font-mono text-xs opacity-40">{s.n}</span>
                  <p className="text-sm font-mono">{s.label}</p>
                </div>
              ))}
            </div>
          </div>

          <div className="relative">
            <img
              src={dashboardPreview}
              alt="Shieldpay dashboard visualizing fraud signal graph"
              width={1280}
              height={1280}
              loading="lazy"
              className="w-full aspect-square object-cover border border-border ring-1 ring-white/10"
            />
            <div className="absolute -bottom-6 -right-4 md:-bottom-10 md:-right-10 w-64 p-6 bg-accent text-white font-mono shadow-2xl shadow-accent/30">
              <div className="text-[10px] uppercase mb-4 opacity-80">
                Detected · anomaly #882
              </div>
              <div className="text-2xl font-black leading-none tracking-tighter">
                FRAUD STOPPED
              </div>
              <div className="mt-4 h-1 w-full bg-white/20 overflow-hidden">
                <div className="h-full w-[86%] bg-white" />
              </div>
              <div className="text-[10px] uppercase mt-2 opacity-70">
                confidence 0.862
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* SIGNALS GRID */}
      <section id="signals" className="py-24 md:py-32 px-6 md:px-10 border-b border-border">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center gap-4 mb-16">
            <span className="font-mono text-xs opacity-40">03 / 04</span>
            <h2 className="text-sm uppercase tracking-[0.3em] font-bold">
              Twenty Signals, One Verdict
            </h2>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-px bg-border border border-border">
            {SIGNALS.map((s) => (
              <div
                key={s.n}
                className="bg-background p-6 hover:bg-white/[0.03] transition-colors group"
              >
                <div className="flex items-center justify-between mb-6">
                  <span className="font-mono text-[10px] opacity-40">#{s.n}</span>
                  <div className="size-1.5 rounded-full bg-border group-hover:bg-accent transition-colors" />
                </div>
                <p className="text-sm font-mono leading-snug">{s.label}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* PRICING */}
      <section id="pricing" className="py-24 md:py-32 px-6 md:px-10 border-b border-border">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center gap-4 mb-16">
            <span className="font-mono text-xs opacity-40">04 / 04</span>
            <h2 className="text-sm uppercase tracking-[0.3em] font-bold">
              Pricing
            </h2>
          </div>
          <div className="mb-16 max-w-2xl">
            <h3 className="text-4xl md:text-5xl font-display italic font-black tracking-tighter">
              Scale with the confidence of a bank.
            </h3>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-px bg-border border border-border">
            {PRICING.map((p) => (
              <div
                key={p.tier}
                className={`p-8 flex flex-col ${
                  p.featured ? "bg-accent text-white" : "bg-background"
                }`}
              >
                <div className={`font-mono text-[10px] uppercase tracking-widest mb-8 ${p.featured ? "opacity-80" : "opacity-40"}`}>
                  {p.featured ? "▸ Recommended" : "Tier"}
                </div>
                <h4 className={`text-2xl font-display italic mb-6 ${p.featured ? "" : ""}`}>{p.tier}</h4>
                <div className="flex items-baseline gap-1 mb-2">
                  <span className="text-5xl font-mono tracking-tighter">{p.price}</span>
                  <span className={`text-xs font-mono ${p.featured ? "opacity-80" : "opacity-50"}`}>
                    {p.per}
                  </span>
                </div>
                <p className={`text-xs font-mono mb-8 ${p.featured ? "opacity-80" : "opacity-50"}`}>
                  {p.note}
                </p>
                <ul className="space-y-3 mb-10 flex-1">
                  {p.features.map((f) => (
                    <li key={f} className="flex items-start gap-3 text-sm font-mono">
                      <span className={p.featured ? "text-white/70" : "text-accent"}>—</span>
                      <span className={p.featured ? "" : "text-muted"}>{f}</span>
                    </li>
                  ))}
                </ul>
                <button
                  onClick={p.featured ? onDemo : undefined}
                  disabled={p.featured && loading}
                  className={`w-full py-3 font-mono text-xs uppercase tracking-widest transition-all disabled:opacity-50 ${
                    p.featured
                      ? "bg-white text-accent hover:brightness-110"
                      : "border border-border hover:bg-white/5"
                  }`}
                >
                  {p.featured && loading ? "Signing in.." : p.cta}
                </button>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* FINAL CTA */}
      <section className="py-32 md:py-40 px-6 md:px-10 text-center relative overflow-hidden">
        <div className="absolute inset-0 pointer-events-none opacity-20">
          <div className="absolute inset-0 bg-[linear-gradient(to_right,#ffffff08_1px,transparent_1px),linear-gradient(to_bottom,#ffffff08_1px,transparent_1px)] bg-[size:6rem_6rem]" />
        </div>
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 size-[500px] rounded-full bg-accent/10 blur-[120px] pointer-events-none" />
        <div className="max-w-4xl mx-auto relative">
          <h2 className="text-6xl md:text-8xl font-display font-black italic tracking-tighter mb-10 leading-none">
            Secure the stream.
          </h2>
          <p className="text-muted max-w-[46ch] mx-auto mb-12 text-lg">
            Deploy Shieldpay in an afternoon. Ship a verdict on your next transaction.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <button
              onClick={onDemo}
              disabled={loading}
              className="px-8 py-4 bg-accent text-white font-mono text-xs uppercase tracking-widest hover:brightness-110 transition-all disabled:opacity-50"
            >
              {loading ? "Signing in.." : "Quick Demo"}
            </button>
            <button className="px-8 py-4 border border-border font-mono text-xs uppercase tracking-widest hover:bg-white/5 transition-all">
              View API Docs
            </button>
          </div>
        </div>
      </section>

      {/* FOOTER */}
      <footer className="py-12 px-6 md:px-10 border-t border-border">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-start gap-8">
          <div className="font-mono text-[10px] tracking-widest uppercase opacity-40">
            © 2026 Shieldpay Corp. All rights reserved.
          </div>
          <div className="flex gap-8 font-mono text-[10px] tracking-widest uppercase opacity-40">
            <a href="#" className="hover:text-accent hover:opacity-100 transition-all">Privacy</a>
            <a href="#" className="hover:text-accent hover:opacity-100 transition-all">Terms</a>
            <a href="#" className="hover:text-accent hover:opacity-100 transition-all">Security</a>
            <a href="#" className="hover:text-accent hover:opacity-100 transition-all">Status</a>
          </div>
        </div>
      </footer>
    </main>
  );
}
