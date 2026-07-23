import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import Header from "./Header.jsx";
import SidebarContent from "./SidebarContent";
import { auth } from "./firebase.js";
import { motion } from "framer-motion";
import { Building2, Loader2, ShieldCheck, AlertTriangle, Search, Download } from "lucide-react";

const FLASK_URL = "http://127.0.0.1:5000";
const MONO_PUBLIC_KEY = import.meta.env.VITE_MONO_PUBLIC_KEY || "";

const ConnectBank = () => {
  const [user, setUser] = useState(null);
  const [connecting, setConnecting] = useState(false);
  const [accountId, setAccountId] = useState(null);
  const [accountDetails, setAccountDetails] = useState(null);
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [fetchingTx, setFetchingTx] = useState(false);
  const [transactions, setTransactions] = useState([]);
  const [fraudResults, setFraudResults] = useState(null);
  const [runningFraud, setRunningFraud] = useState(false);
  const [error, setError] = useState("");
  const [status, setStatus] = useState("");
  const connectRef = useRef(null);

  useEffect(() => {
    const unsub = auth.onAuthStateChanged((u) => {
      setUser(u);
    });
    return () => unsub();
  }, []);

  const launchMonoWidget = () => {
    setError("");
    setStatus("");

    if (!MONO_PUBLIC_KEY) {
      setError("Mono public key not configured. Set VITE_MONO_PUBLIC_KEY in your .env file.");
      return;
    }

    setConnecting(true);

    const customer = user
      ? { email: user.email || "", name: user.displayName || "" }
      : { email: "user@shieldpay.com", name: "ShieldPay User" };

    const connect = new Connect({
      key: MONO_PUBLIC_KEY,
      scope: "auth",
      data: { customer },
      onSuccess: async (data) => {
        const code = data.code;
        setStatus("Account linked! Exchanging token...");
        try {
          const res = await fetch(`${FLASK_URL}/mono/exchange`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ code }),
          });
          const result = await res.json();
          if (res.ok && result.id) {
            setAccountId(result.id);
            setStatus("Bank connected successfully! Account ID: " + result.id);
            fetchAccountDetails(result.id);
          } else {
            setError("Failed to exchange token: " + (result.error || "Unknown error"));
          }
        } catch (err) {
          setError("Backend connection failed: " + err.message);
        }
        setConnecting(false);
      },
      onClose: () => {
        setConnecting(false);
        setStatus("");
      },
      onLoad: () => {
        setStatus("Mono widget loaded...");
      },
    });

    connectRef.current = connect;
    connect.setup();
    connect.open();
  };

  const fetchAccountDetails = async (id) => {
    try {
      const res = await fetch(`${FLASK_URL}/mono/account/${id}`);
      const data = await res.json();
      if (res.ok && data) {
        setAccountDetails(data);
      }
    } catch (err) {
      console.error("Account details error:", err);
    }
  };

  const fetchTransactions = async () => {
    if (!accountId) return;
    setFetchingTx(true);
    setError("");
    setFraudResults(null);

    const params = new URLSearchParams();
    if (startDate) params.set("start", startDate);
    if (endDate) params.set("end", endDate);

    try {
      const res = await fetch(`${FLASK_URL}/mono/transactions/${accountId}?${params.toString()}`);
      const data = await res.json();
      if (res.ok && data.data) {
        setTransactions(data.data);
        setStatus(`Fetched ${data.data.length} transactions`);
      } else if (res.ok && Array.isArray(data)) {
        setTransactions(data);
        setStatus(`Fetched ${data.length} transactions`);
      } else {
        setError("Failed to fetch transactions: " + (data.error || "Unknown error"));
      }
    } catch (err) {
      setError("Backend connection failed: " + err.message);
    }
    setFetchingTx(false);
  };

  const runFraudDetection = async () => {
    if (transactions.length === 0) return;
    setRunningFraud(true);
    setError("");

    try {
      const res = await fetch(`${FLASK_URL}/mono/batch-predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ transactions }),
      });
      const data = await res.json();
      if (res.ok) {
        setFraudResults(data);
        setStatus(`Fraud analysis complete: ${data.fraud_count} suspicious transaction(s) out of ${data.results.length}`);
      } else {
        setError("Fraud detection failed: " + (data.error || "Unknown error"));
      }
    } catch (err) {
      setError("Backend connection failed: " + err.message);
    }
    setRunningFraud(false);
  };

  const formatAmount = (amount) => {
    const val = typeof amount === "number" ? amount : parseFloat(amount || 0);
    return `₦${val.toLocaleString("en-NG", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  };

  const formatDate = (dateStr) => {
    if (!dateStr) return "N/A";
    try {
      return new Date(dateStr).toLocaleDateString("en-NG", {
        year: "numeric",
        month: "short",
        day: "numeric",
      });
    } catch {
      return dateStr;
    }
  };

  const isFraud = (index) => {
    if (!fraudResults) return false;
    const r = fraudResults.results.find((x) => x.index === index);
    return r && r.prediction === 1;
  };

  return (
    <div className="flex min-h-screen bg-[var(--background)] text-[var(--foreground)]">
      <SidebarContent />

      <div className="flex-1 flex flex-col">
        <Header />

        <main className="flex-1 p-6 md:p-8 overflow-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="max-w-5xl mx-auto space-y-6"
          >
            {/* Page Header */}
            <div>
              <h1 className="text-2xl font-bold tracking-tight">Connect Nigerian Bank</h1>
              <p className="text-sm text-[var(--muted)] mt-1">
                Link your bank account via Mono to pull transaction history for real-time fraud analysis.
              </p>
            </div>

            {error && (
              <div className="flex items-center gap-2 rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-400">
                <AlertTriangle className="h-4 w-4 shrink-0" />
                {error}
              </div>
            )}

            {status && !error && (
              <div className="flex items-center gap-2 rounded-lg border border-blue-500/30 bg-blue-500/10 px-4 py-3 text-sm text-blue-400">
                <ShieldCheck className="h-4 w-4 shrink-0" />
                {status}
              </div>
            )}

            {/* Step 1: Connect Bank */}
            <Card className="border-[var(--border)] bg-[var(--card)]">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-base">
                  <Building2 className="h-5 w-5" />
                  Step 1: Link Your Bank Account
                </CardTitle>
              </CardHeader>
              <CardContent>
                {accountId ? (
                  <div className="space-y-3">
                    <div className="flex items-center gap-2 text-sm text-green-400">
                      <ShieldCheck className="h-4 w-4" />
                      Bank connected successfully!
                    </div>
                    {accountDetails && accountDetails.account && (
                      <div className="grid grid-cols-2 gap-3 text-sm">
                        <div>
                          <span className="text-[var(--muted)]">Account Name:</span>{" "}
                          <span className="font-medium">{accountDetails.account.name || "N/A"}</span>
                        </div>
                        <div>
                          <span className="text-[var(--muted)]">Account Number:</span>{" "}
                          <span className="font-medium">{accountDetails.account.accountNumber || "N/A"}</span>
                        </div>
                        <div>
                          <span className="text-[var(--muted)]">Bank:</span>{" "}
                          <span className="font-medium">{accountDetails.account.institution?.name || "N/A"}</span>
                        </div>
                        <div>
                          <span className="text-[var(--muted)]">Balance:</span>{" "}
                          <span className="font-medium">{formatAmount(accountDetails.account.balance)}</span>
                        </div>
                      </div>
                    )}
                    <Button variant="outline" size="sm" onClick={launchMonoWidget} disabled={connecting}>
                      Connect Another Bank
                    </Button>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <p className="text-sm text-[var(--muted)]">
                      Click below to open the Mono Connect widget. You'll select your bank and log in securely.
                      We never see or store your credentials.
                    </p>
                    <Button onClick={launchMonoWidget} disabled={connecting}>
                      {connecting ? (
                        <>
                          <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                          Connecting...
                        </>
                      ) : (
                        <>
                          <Building2 className="h-4 w-4 mr-2" />
                          Connect Your Bank
                        </>
                      )}
                    </Button>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Step 2: Fetch Transactions */}
            {accountId && (
              <Card className="border-[var(--border)] bg-[var(--card)]">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-base">
                    <Search className="h-5 w-5" />
                    Step 2: Fetch Transaction History
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex flex-col sm:flex-row gap-4 items-end">
                    <div className="flex-1 space-y-1.5">
                      <Label htmlFor="start-date">Start Date</Label>
                      <Input
                        id="start-date"
                        type="date"
                        value={startDate}
                        onChange={(e) => setStartDate(e.target.value)}
                      />
                    </div>
                    <div className="flex-1 space-y-1.5">
                      <Label htmlFor="end-date">End Date</Label>
                      <Input
                        id="end-date"
                        type="date"
                        value={endDate}
                        onChange={(e) => setEndDate(e.target.value)}
                      />
                    </div>
                    <Button onClick={fetchTransactions} disabled={fetchingTx}>
                      {fetchingTx ? (
                        <>
                          <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                          Fetching...
                        </>
                      ) : (
                        <>
                          <Download className="h-4 w-4 mr-2" />
                          Fetch Transactions
                        </>
                      )}
                    </Button>
                  </div>
                  <p className="text-xs text-[var(--muted)] mt-2">
                    Leave dates empty to fetch the most recent transactions.
                  </p>
                </CardContent>
              </Card>
            )}

            {/* Step 3: Fraud Detection */}
            {transactions.length > 0 && (
              <Card className="border-[var(--border)] bg-[var(--card)]">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-base">
                    <ShieldCheck className="h-5 w-5" />
                    Step 3: Run Fraud Detection
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center justify-between mb-4">
                    <p className="text-sm text-[var(--muted)]">
                      {transactions.length} transactions loaded. Run the AI model to flag suspicious activity.
                    </p>
                    <Button onClick={runFraudDetection} disabled={runningFraud}>
                      {runningFraud ? (
                        <>
                          <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                          Analyzing...
                        </>
                      ) : (
                        <>
                          <ShieldCheck className="h-4 w-4 mr-2" />
                          Analyze for Fraud
                        </>
                      )}
                    </Button>
                  </div>

                  {fraudResults && (
                    <div className="mb-4 flex items-center gap-4">
                      <div className="flex items-center gap-2 rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-2 text-sm">
                        <AlertTriangle className="h-4 w-4 text-red-400" />
                        <span className="text-red-400 font-medium">{fraudResults.fraud_count} Flagged</span>
                      </div>
                      <div className="flex items-center gap-2 rounded-lg border border-green-500/30 bg-green-500/10 px-4 py-2 text-sm">
                        <ShieldCheck className="h-4 w-4 text-green-400" />
                        <span className="text-green-400 font-medium">
                          {fraudResults.results.length - fraudResults.fraud_count} Clean
                        </span>
                      </div>
                    </div>
                  )}

                  {/* Transactions Table */}
                  <div className="rounded-lg border border-[var(--border)] overflow-hidden">
                    <div className="max-h-[400px] overflow-y-auto">
                      <table className="w-full text-sm">
                        <thead className="sticky top-0 bg-[var(--card)] border-b border-[var(--border)]">
                          <tr className="text-left text-xs uppercase tracking-wider text-[var(--muted)]">
                            <th className="px-4 py-3">Date</th>
                            <th className="px-4 py-3">Description</th>
                            <th className="px-4 py-3">Type</th>
                            <th className="px-4 py-3 text-right">Amount</th>
                            {fraudResults && <th className="px-4 py-3 text-center">Status</th>}
                          </tr>
                        </thead>
                        <tbody>
                          {transactions.map((tx, i) => (
                            <tr
                              key={i}
                              className={`border-b border-[var(--border)] hover:bg-white/[0.02] ${
                                isFraud(i) ? "bg-red-500/5" : ""
                              }`}
                            >
                              <td className="px-4 py-3 text-[var(--muted)]">{formatDate(tx.date)}</td>
                              <td className="px-4 py-3">{tx.narration || tx.description || "—"}</td>
                              <td className="px-4 py-3">
                                <span
                                  className={`px-2 py-0.5 rounded text-xs font-medium ${
                                    (tx.type || "").toLowerCase() === "credit"
                                      ? "bg-green-500/10 text-green-400"
                                      : "bg-orange-500/10 text-orange-400"
                                  }`}
                                >
                                  {tx.type || "N/A"}
                                </span>
                              </td>
                              <td className="px-4 py-3 text-right font-mono">{formatAmount(tx.amount)}</td>
                              {fraudResults && (
                                <td className="px-4 py-3 text-center">
                                  {isFraud(i) ? (
                                    <span className="inline-flex items-center gap-1 text-red-400 text-xs font-medium">
                                      <AlertTriangle className="h-3 w-3" />
                                      Fraud
                                    </span>
                                  ) : (
                                    <span className="inline-flex items-center gap-1 text-green-400 text-xs font-medium">
                                      <ShieldCheck className="h-3 w-3" />
                                      Clean
                                    </span>
                                  )}
                                </td>
                              )}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </motion.div>
        </main>
      </div>
    </div>
  );
};

export default ConnectBank;
