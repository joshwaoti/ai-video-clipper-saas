// No longer a client component: removed "use client";

import { redirect } from "next/navigation";
import { BillingClient } from "~/components/billing-client"; // Import the new client component
import { auth } from "~/server/auth";
import { db } from "~/server/db";

// PricingPlan interface, plans array, and PricingCard component are removed from here.

export default async function BillingPage() {
  const session = await auth();

  if (!session?.user?.id) {
    redirect("/login"); // Redirect to login if no session
  }

  const userData = await db.user.findUniqueOrThrow({
    where: { id: session.user.id },
    select: {
      phoneNumber: true,
      credits: {
        select: {
          amount: true,
        },
      },
    },
  });

  return (
    <BillingClient
      userPhoneNumber={userData.phoneNumber}
      initialCredits={userData.credits?.amount ?? 0}
    />
  );
}
