import { NextResponse } from "next/server";

export async function GET() {
  return NextResponse.json({
    message: "Callback URL is accessible",
    timestamp: new Date().toISOString(),
  });
}
