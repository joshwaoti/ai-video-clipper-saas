# RevelationsHub - Product Requirements Document

## Executive Summary

RevelationsHub is a comprehensive "Ministry Operating System" that unifies media engineering, pastoral care, and content dissemination for faith-based organizations. Built on top of the existing ai-podcast-clipper-backend, it targets the sermon repurposing market currently led by SermonShots.

## Design Philosophy: "Divine Harmony"

### Visual Identity
A sophisticated palette distinguishing RevelationsHub from competitors:
- **Dark Amethyst** (#301a4b) - Base background, replacing typical black/slate
- **Pacific Blue** (#6db1bf) - Primary actions, CTAs, active states
- **Cotton Candy** (#f39a9d) - AI features, highlights
- **Lavender Blush** (#ffeaec) - Light mode backgrounds, text on dark
- **Hunter Green** (#3f6c51) - Success states, ready badges

### Dual Theme System
- **Sanctuary Mode (Dark)**: Default for media-heavy tasks (dashboard, video editing)
- **Scripture Mode (Light)**: For text-heavy tasks (blogs, discussion guides)

## Tech Stack

| Technology | Purpose |
|-----------|---------|
| Next.js 15 | React framework with App Router |
| Clerk | Authentication & user management |
| Convex | Real-time database |
| Inngest | Background job processing |
| Paystack | Payment processing |
| PostHog | Product analytics |
| Tailwind CSS v4 | Styling |

## Core Features

### 1. Marketing Site
- Landing page with hero section
- Feature bento grid (Magic Clips, Discussion Guides, Social Carousel)
- Pricing page (Plus, Silver, Gold tiers)

### 2. Authentication
- Sign up / Sign in (via Clerk)
- Onboarding flow (Organization Profile, Brand Kit Setup)

### 3. Sermon Library
- Grid view of uploaded sermons
- Status badges (Ready, Processing)
- Search and filter functionality

### 4. Sermon Dashboard (Hub-and-Spoke)
Single sidebar UX with tools grouped by category:
- **Social Media**: Magic Clips, Image Quotes, Social Carousel
- **Discipleship**: Discussion Guides, Devotionals, Sermon Outline
- **More Content**: Transcription, Blog Post, Podcast Audio, Summaries

### 5. Video Editor
- Timeline with video, audio waveform, and caption tracks
- Export functionality

### 6. User Brand Kit
- Logo upload (light/dark versions)
- Custom color palette per church
- Font selection

## Target Users

| Role | Permissions |
|------|-------------|
| Admin | Full access (billing, team, brand kit) |
| Editor | Upload, edit, generate content |
| Contributor | View and download assets |

## Pricing Tiers (Planned)

| Tier | Features |
|------|----------|
| Free | 2 clips, watermarked |
| Plus | Unlimited clips, AI Camera |
| Silver | Discipleship tools, Blog Generator |
| Gold | Advanced text manipulation |
| Platinum | Live translation |

## Backend Integration

The existing `ai-podcast-clipper-backend` provides:
- Video processing via Modal (serverless GPU)
- WhisperX transcription
- Gemini AI for moment identification
- S3 storage for clips
- Supports podcast and sermon modes

## Development Phases

1. **Phase 1**: UI-only implementation (no backend integration)
2. **Phase 2**: Clerk authentication integration
3. **Phase 3**: Convex database setup
4. **Phase 4**: Backend API integration
5. **Phase 5**: Paystack billing
6. **Phase 6**: Inngest background jobs
7. **Phase 7**: PostHog analytics

<!-- npx inngest-cli@latest dev -->