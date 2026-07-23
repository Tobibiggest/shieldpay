import { createFileRoute } from "@tanstack/react-router";
import Dashboard from "@/components/logic/Dashboard.jsx";

export const Route = createFileRoute("/dashboard")({
  component: Dashboard,
});
