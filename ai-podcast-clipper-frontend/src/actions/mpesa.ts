"use server";

import { redirect } from "next/navigation";
import { env } from "~/env";
import { auth } from "~/server/auth";
import { db } from "~/server/db";
import { MpesaService } from "~/lib/mpesa/service";

export type CreditPack = "small" | "medium" | "large";

const CREDIT_PACKS: Record<CreditPack, { amount: number; credits: number }> = {
  small: { amount: 999, credits: 50 }, // KES 999
  medium: { amount: 2499, credits: 150 }, // KES 2,499
  large: { amount: 6999, credits: 500 }, // KES 6,999
};

export async function initiateMpesaPayment(creditPack: CreditPack) {
  const serverSession = await auth();
  if (!serverSession?.user.id) {
    throw new Error("User not authenticated");
  }

  const user = await db.user.findUniqueOrThrow({
    where: { id: serverSession.user.id },
    select: { phoneNumber: true },
  });

  if (!user.phoneNumber) {
    throw new Error("Phone number is required for MPESA payment");
  }

  const mpesaService = new MpesaService();
  const pack = CREDIT_PACKS[creditPack];

  try {
    const response = await mpesaService.initiateSTKPush(
      user.phoneNumber,
      pack.amount,
      `Credit Pack ${creditPack}`,
      `Purchase of ${pack.credits} credits`,
    );

    if (!response.CheckoutRequestID || !response.MerchantRequestID) {
      throw new Error("Invalid response from MPESA");
    }

    // Store the transaction details in the database
    await db.transaction.create({
      data: {
        userId: serverSession.user.id,
        amount: pack.amount,
        credits: pack.credits,
        status: "pending",
        checkoutRequestId: response.CheckoutRequestID,
        merchantRequestId: response.MerchantRequestID,
      },
    });

    return { success: true, message: response.CustomerMessage };
  } catch (error) {
    console.error("MPESA payment initiation error:", error);
    return {
      success: false,
      error:
        error instanceof Error ? error.message : "Failed to initiate payment",
    };
  }
}
