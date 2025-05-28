import { NextResponse } from "next/server";
import { db } from "~/server/db";
import { Prisma } from "@prisma/client";

interface RequestBody {
  transactionId: string;
}

export async function POST(request: Request) {
  try {
    const { transactionId } = (await request.json()) as RequestBody;

    const transaction = await db.transaction.findUnique({
      where: { id: transactionId },
    });

    if (!transaction) {
      return NextResponse.json(
        { error: "Transaction not found" },
        { status: 404 },
      );
    }

    // Check if transaction has already been processed
    if (transaction.status === "completed") {
      return NextResponse.json(
        { error: "Transaction has already been processed" },
        { status: 400 },
      );
    }

    // Update transaction status
    await db.transaction.update({
      where: { id: transaction.id },
      data: {
        status: "completed",
      },
    });

    // Add credits to user's account
    const updatedUser = await db.user.update({
      where: { id: transaction.userId },
      data: {
        credits: {
          increment: transaction.credits,
        },
      },
    });

    return NextResponse.json({
      success: true,
      message: "Credits added successfully",
      newBalance: updatedUser.credits,
    });
  } catch (error) {
    console.error("Manual Update Error:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 },
    );
  }
}
