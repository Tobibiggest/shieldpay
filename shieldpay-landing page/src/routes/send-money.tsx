import { createFileRoute } from "@tanstack/react-router";
import Homepage from "@/components/logic/homepage.jsx";

export const Route = createFileRoute("/send-money")({
  component: Homepage,
});
