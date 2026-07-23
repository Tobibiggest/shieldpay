import { createFileRoute } from "@tanstack/react-router";
import ConnectBank from "@/components/logic/ConnectBank.jsx";

export const Route = createFileRoute("/connect-bank")({
  component: ConnectBank,
});
