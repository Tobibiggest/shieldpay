import { createFileRoute } from "@tanstack/react-router";
import RecentTransactions from "@/components/logic/Recent.jsx";

export const Route = createFileRoute("/transactions")({
  component: RecentTransactions,
});
