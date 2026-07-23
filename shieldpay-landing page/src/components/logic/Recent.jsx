"use client"

import React, { useState, useEffect } from "react"
import { motion } from "framer-motion"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Search, ArrowUpRight, ArrowDownLeft } from 'lucide-react'
import Header from "./Header"
import SidebarContent from "./SidebarContent"
import { db } from './firebase' // Import your Firebase configuration
import { collection, getDocs, query, where, doc, getDoc } from 'firebase/firestore'
import { auth } from './firebase' // Import Firebase auth

const RecentTransactions = () => {
  const [user, setUser] = useState(null)
  const [searchTerm, setSearchTerm] = useState("")
  const [transactions, setTransactions] = useState([]) // State to hold transactions

  useEffect(() => {
    const fetchUserData = async () => {
      const currentUser = auth.currentUser // Get the current user
      if (currentUser) {
        const userRef = doc(db, "users", currentUser.uid) // Reference to the user's document
        const userDoc = await getDoc(userRef) // Fetch the user document
        if (userDoc.exists()) {
          setUser(userDoc.data()) // Set user data
        } else {
          console.error("User document does not exist")
        }
      } else {
        console.log("No user is currently logged in")
      }
    }

    const fetchTransactions = async () => {
      if (!user) return // Ensure user is defined

      const transactionsCollection = collection(db, "transactions")
      const transactionsQuery = query(transactionsCollection, where("senderUPI", "==", user.upiId)) // Fetch transactions for the current user's UPI ID
      const transactionSnapshot = await getDocs(transactionsQuery)
      const transactionList = transactionSnapshot.docs.map(doc => ({
        id: doc.id,
        ...doc.data()
      }))
      setTransactions(transactionList)
    }

    fetchUserData().then(fetchTransactions) // Fetch user data and then transactions
  }, [user]) // Dependency on user

  const filteredTransactions = transactions.filter(
    (transaction) =>
      transaction.recipientUPI.toLowerCase().includes(searchTerm.toLowerCase()) ||
      transaction.amount.toString().includes(searchTerm) ||
      transaction.remarks.toLowerCase().includes(searchTerm.toLowerCase())
  )

  return (
    <div className="flex min-h-screen" style={{background:'var(--background)', color:'var(--foreground)'}}>
      <aside className="hidden md:flex flex-col w-60 min-h-screen border-r border-[var(--border)]">
        <SidebarContent />
      </aside>

      <div className="flex-1 flex flex-col">
        <Header user={user} />

        <div className="flex-1 p-8">
          {/* Page heading + search */}
          <div className="flex items-center justify-between mb-8">
            <div>
              <div className="font-mono text-[10px] uppercase tracking-widest opacity-40 mb-1">03 / Ledger</div>
              <h1 className="font-display italic text-3xl font-black">Recent Transactions</h1>
            </div>
            <div className="relative">
              <Search className="absolute left-3 top-2.5 h-3.5 w-3.5 opacity-40" />
              <input
                type="text"
                placeholder="Search..."
                className="pl-9 pr-4 py-2 w-52 bg-white/5 border border-[var(--border)] font-mono text-[11px] placeholder:opacity-40 focus:outline-none focus:border-[var(--accent)] transition-colors"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>
          </div>

          {/* Transaction list */}
          <div className="border border-[var(--border)] divide-y divide-[var(--border)]">
            {/* Table header */}
            <div className="grid grid-cols-12 px-6 py-3 font-mono text-[10px] uppercase tracking-widest opacity-40">
              <span className="col-span-1">#</span>
              <span className="col-span-4">Recipient</span>
              <span className="col-span-3">Date</span>
              <span className="col-span-2">Status</span>
              <span className="col-span-2 text-right">Amount</span>
            </div>
            {filteredTransactions.map((transaction, i) => (
              <div
                key={transaction.id}
                className="grid grid-cols-12 px-6 py-4 items-center hover:bg-white/[0.02] transition-colors group"
              >
                <span className="col-span-1 font-mono text-[10px] opacity-30">{String(i + 1).padStart(2, '0')}</span>
                <div className="col-span-4 flex items-center gap-3">
                  <div className={`w-1.5 h-1.5 rounded-full shrink-0 ${transaction.type === 'incoming' ? '' : ''}`}
                    style={{background: transaction.type === 'incoming' ? 'var(--accent)' : '#ef4444'}} />
                  <span className="font-mono text-xs truncate">{transaction.recipientUPI}</span>
                </div>
                <span className="col-span-3 font-mono text-[11px] opacity-50">
                  {new Date(transaction.createdAt.seconds * 1000).toLocaleDateString()}
                </span>
                <span className={`col-span-2 font-mono text-[10px] uppercase tracking-widest ${
                  transaction.status === 'Completed' ? '' : 'opacity-50'
                }`} style={{color: transaction.status === 'Completed' ? 'var(--accent)' : undefined}}>
                  {transaction.status}
                </span>
                <span className={`col-span-2 text-right font-mono text-sm font-medium`}
                  style={{color: transaction.type === 'incoming' ? 'var(--accent)' : '#ef4444'}}>
                  {transaction.type === 'incoming' ? '+' : '-'}₦{transaction.amount.toFixed(2)}
                </span>
              </div>
            ))}
            {filteredTransactions.length === 0 && (
              <div className="px-6 py-12 text-center font-mono text-[11px] uppercase tracking-widest opacity-30">
                No transactions found
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default RecentTransactions

