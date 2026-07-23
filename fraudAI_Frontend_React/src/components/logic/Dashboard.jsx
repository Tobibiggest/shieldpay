import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { auth, db } from "./firebase.js";
import { signOut } from "firebase/auth";
import { doc, getDoc } from "firebase/firestore";
import Header from "./Header.jsx";
import SidebarContent from "./SidebarContent";
import { handleGoogleSignIn } from "./auth";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { DollarSign, CreditCard, Activity, Zap } from 'lucide-react';
import { motion } from "framer-motion";
import { Line, LineChart, PieChart, Pie, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis, Legend } from 'recharts';

const transactionData = [
  { name: 'Jan', value: 400 },
  { name: 'Feb', value: 300 },
  { name: 'Mar', value: 600 },
  { name: 'Apr', value: 800 },
  { name: 'May', value: 500 },
  { name: 'Jun', value: 700 },
];

const spendingData = [
  { name: 'Food', value: 400 },
  { name: 'Transport', value: 300 },
  { name: 'Shopping', value: 300 },
  { name: 'Bills', value: 200 },
];

const COLORS = ['#3b82f6', '#60a5fa', '#93c5fd', '#1d4ed8'];

const Dashboard = () => {
  const [user, setUser] = useState(null);
  const [upiId, setUpiId] = useState("");
  const [balance, setBalance] = useState(0);

  const handleSignOut = async () => {
    try {
      await signOut(auth);
      setUser(null);
      setUpiId("");
    } catch (error) {
      console.error("Sign-Out Error:", error);
    }
  };

  useEffect(() => {
   
    const checkUser = async () => {
      const currentUser = auth.currentUser;
      if (currentUser) {
        setUser(currentUser);
        const userRef = doc(db, "users", currentUser.uid);
        const userDoc = await getDoc(userRef);
        if (userDoc.exists()) {
          setUpiId(userDoc.data().upiId);
          setBalance(Math.floor(Math.random() * 10000));
        }
      }
    };
    checkUser();
  }, []);

  const TransactionChart = () => (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={transactionData}>
        <XAxis 
          dataKey="name" 
          stroke="#888888" 
          fontSize={12} 
          tickLine={false} 
          axisLine={false}
        />
        <YAxis
          stroke="#888888"
          fontSize={12}
          tickLine={false}
          axisLine={false}
          tickFormatter={(value) => `₦${value}`}
        />
        <Tooltip
          content={({ active, payload }) => {
            if (active && payload && payload.length) {
              return (
                <div className="border border-[var(--border)] p-2" style={{background:'var(--background)'}}>
                  <p className="font-mono text-[11px]" style={{color:'var(--accent)'}}>{`₦${payload[0].value}`}</p>
                </div>
              );
            }
            return null;
          }}
        />
        <Line
          type="monotone"
          dataKey="value"
          stroke="#3b82f6"
          strokeWidth={2}
          dot={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );

  const SpendingPieChart = () => (
    <ResponsiveContainer width="100%" height={300}>
      <PieChart>
        <Pie
          data={spendingData}
          cx="50%"
          cy="50%"
          labelLine={false}
          outerRadius={80}
          fill="#8884d8"
          dataKey="value"
        >
          {spendingData.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
          ))}
        </Pie>
        <Tooltip
          content={({ active, payload }) => {
            if (active && payload && payload.length) {
              return (
                <div className="border border-[var(--border)] p-2" style={{background:'var(--background)'}}>
                  <p className="font-mono text-[11px]" style={{color:'var(--accent)'}}>{`${payload[0].name}: ₦${payload[0].value}`}</p>
                </div>
              );
            }
            return null;
          }}
        />
        <Legend 
          formatter={(value, entry, index) => <span className="text-gray-400">{value}</span>}
        />
      </PieChart>
    </ResponsiveContainer>
  );

  return (
    <div className="flex min-h-screen" style={{background:'var(--background)', color:'var(--foreground)'}}>
      <aside className="hidden md:flex flex-col w-60 min-h-screen border-r border-[var(--border)]">
        <SidebarContent />
      </aside>
      <div className="flex-1 flex flex-col overflow-y-auto">
        <Header user={user} onSignIn={handleGoogleSignIn} />
        <div className="flex-1 p-8">
          {/* User row */}
          <div className="flex items-center justify-between mb-10">
            <div className="flex items-center gap-4">
              <Avatar className="h-10 w-10 border border-[var(--border)]">
                <AvatarImage src={user?.photoURL} alt={user?.displayName} />
                <AvatarFallback className="font-mono text-sm" style={{background:'var(--accent)', color:'#fff'}}>{user?.displayName?.charAt(0)}</AvatarFallback>
              </Avatar>
              <div>
                <div className="font-mono text-xs uppercase tracking-widest opacity-40 mb-0.5">Operator</div>
                <div className="font-display italic text-lg font-black">{user?.displayName}</div>
                <div className="font-mono text-[10px] opacity-40">{upiId}</div>
              </div>
            </div>
            <button
              onClick={handleSignOut}
              className="px-4 py-2 border border-[var(--border)] font-mono text-[10px] uppercase tracking-widest opacity-60 hover:opacity-100 hover:border-[var(--foreground)] transition-all"
            >
              Sign Out
            </button>
          </div>

          {/* Stat cards */}
          <div className="grid gap-px md:grid-cols-2 lg:grid-cols-4 border border-[var(--border)] mb-8" style={{background:'var(--border)'}}>
            {[
              { tag: '01', title: 'Total Balance',       value: `₦${balance.toFixed(2)}` },
              { tag: '02', title: 'Monthly Spending',    value: `₦${(balance * 0.3).toFixed(2)}` },
              { tag: '03', title: 'Total Transactions',  value: transactionData.length },
              { tag: '04', title: 'Cashback Earned',     value: `₦${(balance * 0.02).toFixed(2)}` },
            ].map((item) => (
              <div key={item.tag} className="p-6" style={{background:'var(--background)'}}>
                <div className="font-mono text-[10px] opacity-40 mb-4 uppercase tracking-widest">[{item.tag}] {item.title}</div>
                <div className="text-3xl font-mono tracking-tight" style={{color:'var(--accent)'}}>{item.value}</div>
              </div>
            ))}
          </div>

          {/* Charts */}
          <div className="grid gap-8 md:grid-cols-2">
            <div className="border border-[var(--border)] p-6">
              <div className="font-mono text-[10px] uppercase tracking-widest opacity-40 mb-6">Transaction History</div>
              <TransactionChart />
            </div>
            <div className="border border-[var(--border)] p-6">
              <div className="font-mono text-[10px] uppercase tracking-widest opacity-40 mb-6">Spending Categories</div>
              <SpendingPieChart />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;

