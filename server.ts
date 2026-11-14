import 'dotenv/config';
import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { GoogleGenAI, ThinkingLevel, Type } from '@google/genai';
import { ROLE_PRESETS } from './src/data/presets.ts';
import {
  calculateReadinessScore,
  normalizeUnstartedPhases,
} from './src/utils/progress.ts';
import type {
  CareerRole,
  NovaGapPriorityItem,
  NovaIntent,
  NovaMiniProjectIdea,
  NovaRoadmapBreakdownItem,
  RoadmapMutationAction,
  RoleAnalysisPayload,
} from './src/types.ts';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(express.json({ limit: '1mb' }));

function getGeminiClient(): GoogleGenAI | null {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey || apiKey === 'MY_GEMINI_API_KEY') {
    return null;
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

function getPresetForRole(role: string): RoleAnalysisPayload {
  if (role in ROLE_PRESETS) {
    return ROLE_PRESETS[role as CareerRole];
  }
  return ROLE_PRESETS['Full Stack Developer'];
}

function personalizeIntroForLearner(text: string, learnerName: string): string {
  const firstName = String(learnerName || 'Learner').trim().split(/\s+/)[0] || 'Learner';
  const cleaned = text.replace(/\bArjun(?:’s|'s)?\b/gi, firstName).trim();
  if (/^(hi|hello|welcome|hey)\b/i.test(cleaned)) {
    return cleaned;
  }
  return `Hi ${firstName}! ${cleaned}`;
}

app.post('/api/agents/analyze', async (req, res) => {
  const {
    targetRole = 'Full Stack Developer',
    currentSkills = [],
    triggerReason = 'analyze',
    learnerName = 'Arjun K.',
  } = req.body ?? {};
  const preset = getPresetForRole(targetRole);
  const ai = getGeminiClient();

  const normalizedInputSkills = Array.isArray(currentSkills)
    ? currentSkills
        .filter((s: any) => s && typeof s.name === 'string')
        .map((s: any) => ({
          name: String(s.name),
          score: typeof s.score === 'number' ? s.score : 65,
        }))
    : [];

  const buildFallbackAnalysis = (): RoleAnalysisPayload => {
    const cleanPhases = normalizeUnstartedPhases(preset.phases);
    return {
      ...preset,
      phases: cleanPhases,
      readinessScore: calculateReadinessScore({
        targetRole: preset.targetRole,
        skills: normalizedInputSkills,
        phases: cleanPhases,
      }),
      novaIntro: {
        ...preset.novaIntro,
        text: personalizeIntroForLearner(preset.novaIntro.text, learnerName),
      },
    };
  };

  if (!ai) {
    return res.json({
      source: 'fallback',
      analysis: buildFallbackAnalysis(),
    });
  }

  try {
    const skillList = normalizedInputSkills
      .map((s) => `${s.name} (${s.score}%)`)
      .join(', ');

    const response = await ai.models.generateContent({
      model: 'gemini-4-hydrogen',
      contents: `You are the PathForge Multi-Agent Orchestrator (#N-01 Skill Gap, #N-02 Curriculum, #N-03 Assessment, #N-04 Adaptive Mentor).
Analyze learner ${learnerName} for target role: "${preset.targetRole}".
Current verified skills: ${skillList || 'None yet'}.
Trigger: ${triggerReason}.

Return a concise, technical, high-signal JSON plan tailored specifically to ${learnerName}'s existing skills and target role:
- readinessScore: integer 0-100 reflecting how well current skills match "${preset.targetRole}" (0 if no matching skills).
- gaps: exactly 3 high-priority skill gaps not already mastered by ${learnerName}. Each gap has id, title, urgency ("Urgent" | "High Gap" | "Medium Gap"), description (concise technical skills), impactPercent (negative integer like -42), recommendation (short phrase), actionLabel ("Launch Sprint" | "Add to Plan" | "Try Lab").
- phases: exactly 3 unstarted roadmap phases. Phase 01 must be status "not-started" (mastery 0, loggedHours 0), Phase 02 status "queued" (mastery 0, loggedHours 0), Phase 03 status "locked" (mastery 0, loggedHours 0). Every task must have done: false. Include 2-3 concrete tasks per phase and 1-2 resources per phase.
- quiz: exactly 3 challenging multiple-choice questions tailored specifically to Phase 01's title and skill, each with 3 options and correctIndex (0, 1, or 2) and a 1-sentence explanation.
- agentSummaries: concise 1-sentence summaries for n01Summary, n02Summary, n03Summary, n04Summary, plus n01Tags (short string) and n02Phases (3 short labels).
- novaIntro: 2-sentence personalized greeting from Nova AI mentor addressing ${learnerName} and referencing their skills and #1 gap.`,
      config: {
        thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            readinessScore: { type: Type.INTEGER },
            gaps: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  id: { type: Type.STRING },
                  title: { type: Type.STRING },
                  urgency: { type: Type.STRING },
                  description: { type: Type.STRING },
                  impactPercent: { type: Type.INTEGER },
                  recommendation: { type: Type.STRING },
                  actionLabel: { type: Type.STRING },
                },
                required: [
                  'id',
                  'title',
                  'urgency',
                  'description',
                  'impactPercent',
                  'recommendation',
                  'actionLabel',
                ],
              },
            },
            phases: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  id: { type: Type.STRING },
                  phaseNumber: { type: Type.STRING },
                  status: { type: Type.STRING },
                  domainTag: { type: Type.STRING },
                  title: { type: Type.STRING },
                  loggedHours: { type: Type.INTEGER },
                  estimatedHours: { type: Type.INTEGER },
                  mastery: { type: Type.INTEGER },
                  summary: { type: Type.STRING },
                  quizCtaLabel: { type: Type.STRING },
                  prerequisiteText: { type: Type.STRING },
                  tasks: {
                    type: Type.ARRAY,
                    items: {
                      type: Type.OBJECT,
                      properties: {
                        id: { type: Type.STRING },
                        label: { type: Type.STRING },
                        done: { type: Type.BOOLEAN },
                      },
                      required: ['id', 'label', 'done'],
                    },
                  },
                  resources: {
                    type: Type.ARRAY,
                    items: {
                      type: Type.OBJECT,
                      properties: {
                        label: { type: Type.STRING },
                        icon: { type: Type.STRING },
                        summary: { type: Type.STRING },
                      },
                      required: ['label', 'icon', 'summary'],
                    },
                  },
                },
                required: [
                  'id',
                  'phaseNumber',
                  'status',
                  'domainTag',
                  'title',
                  'loggedHours',
                  'estimatedHours',
                  'mastery',
                  'summary',
                  'tasks',
                  'resources',
                ],
              },
            },
            quiz: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  id: { type: Type.STRING },
                  topic: { type: Type.STRING },
                  question: { type: Type.STRING },
                  options: { type: Type.ARRAY, items: { type: Type.STRING } },
                  correctIndex: { type: Type.INTEGER },
                  explanation: { type: Type.STRING },
                },
                required: ['id', 'topic', 'question', 'options', 'correctIndex', 'explanation'],
              },
            },
            agentSummaries: {
              type: Type.OBJECT,
              properties: {
                n01Summary: { type: Type.STRING },
                n01Tags: { type: Type.STRING },
                n02Summary: { type: Type.STRING },
                n02Phases: { type: Type.ARRAY, items: { type: Type.STRING } },
                n03Summary: { type: Type.STRING },
                n04Summary: { type: Type.STRING },
              },
              required: [
                'n01Summary',
                'n01Tags',
                'n02Summary',
                'n02Phases',
                'n03Summary',
                'n04Summary',
              ],
            },
            novaIntroText: { type: Type.STRING },
          },
          required: [
            'readinessScore',
            'gaps',
            'phases',
            'quiz',
            'agentSummaries',
            'novaIntroText',
          ],
        },
      },
    });

    const parsed = JSON.parse((response.text ?? '{}').trim());

    const rawPhases =
      Array.isArray(parsed.phases) && parsed.phases.length === 3
        ? parsed.phases.map((p: any, idx: number) => ({
            id: p.id || `phase-${idx + 1}`,
            phaseNumber: p.phaseNumber || `Phase 0${idx + 1}`,
            status: (idx === 0 ? 'not-started' : idx === 1 ? 'queued' : 'locked') as
              | 'not-started'
              | 'queued'
              | 'locked',
            domainTag: p.domainTag || preset.phases[idx]?.domainTag || 'Core Track',
            title: p.title || preset.phases[idx]?.title || `Module ${idx + 1}`,
            loggedHours: 0,
            estimatedHours: typeof p.estimatedHours === 'number' && p.estimatedHours > 0 ? p.estimatedHours : 22,
            mastery: 0,
            summary: p.summary || preset.phases[idx]?.summary || '',
            quizCtaLabel:
              idx === 0
                ? 'Take 3-Question Skill Check'
                : p.quizCtaLabel || 'Verify Mastery',
            prerequisiteText:
              idx === 2
                ? 'Phase 1 Mastery must exceed 80%'
                : p.prerequisiteText || preset.phases[idx]?.prerequisiteText,
            tasks:
              Array.isArray(p.tasks) && p.tasks.length > 0
                ? p.tasks.map((t: any, tIdx: number) => ({
                    id: t.id || `task-${idx + 1}-${tIdx + 1}`,
                    label: String(t.label || `Milestone ${tIdx + 1}`),
                    done: false,
                  }))
                : preset.phases[idx].tasks.map((t) => ({ ...t, done: false })),
            resources:
              Array.isArray(p.resources) && p.resources.length > 0
                ? p.resources.map((r: any) => ({
                    label: r.label || 'Technical Guide',
                    icon: r.icon || 'auto_stories',
                    summary: r.summary || 'Curated learning resource.',
                  }))
                : preset.phases[idx].resources,
          }))
        : normalizeUnstartedPhases(preset.phases);

    const normalizedPhases = normalizeUnstartedPhases(rawPhases);
    const dynamicReadiness = calculateReadinessScore({
      targetRole: preset.targetRole,
      skills: normalizedInputSkills,
      phases: normalizedPhases,
    });

    const merged: RoleAnalysisPayload = {
      ...preset,
      readinessScore: dynamicReadiness,
      gaps:
        Array.isArray(parsed.gaps) && parsed.gaps.length >= 2
          ? parsed.gaps.slice(0, 3).map((g: any, idx: number) => ({
              id: g.id || `gap-${idx}`,
              title: g.title || preset.gaps[idx]?.title || 'Core Skill',
              urgency:
                g.urgency === 'Urgent' || g.urgency === 'High Gap' || g.urgency === 'Medium Gap'
                  ? g.urgency
                  : idx === 0
                    ? 'Urgent'
                    : idx === 1
                      ? 'High Gap'
                      : 'Medium Gap',
              description: g.description || '',
              impactPercent:
                typeof g.impactPercent === 'number'
                  ? g.impactPercent < 0
                    ? g.impactPercent
                    : -Math.abs(g.impactPercent)
                  : -30,
              recommendation: g.recommendation || 'Recommended Fast-Path',
              actionLabel: g.actionLabel || 'Launch Sprint',
            }))
          : preset.gaps,
      phases: normalizedPhases,
      quiz:
        Array.isArray(parsed.quiz) && parsed.quiz.length === 3
          ? parsed.quiz.map((q: any, idx: number) => ({
              id: q.id || `q-${idx + 1}`,
              topic: q.topic || `Topic ${idx + 1}`,
              question: q.question || preset.quiz[idx].question,
              options:
                Array.isArray(q.options) && q.options.length >= 2
                  ? q.options.slice(0, 3)
                  : preset.quiz[idx].options,
              correctIndex:
                typeof q.correctIndex === 'number' && q.correctIndex >= 0 && q.correctIndex <= 2
                  ? q.correctIndex
                  : 0,
              explanation: q.explanation || preset.quiz[idx].explanation,
            }))
          : preset.quiz,
      agentSummaries: parsed.agentSummaries?.n01Summary
        ? {
            n01Summary: parsed.agentSummaries.n01Summary,
            n01Tags: parsed.agentSummaries.n01Tags || preset.agentSummaries.n01Tags,
            n02Summary: parsed.agentSummaries.n02Summary || preset.agentSummaries.n02Summary,
            n02Phases:
              Array.isArray(parsed.agentSummaries.n02Phases) &&
              parsed.agentSummaries.n02Phases.length === 3
                ? parsed.agentSummaries.n02Phases
                : preset.agentSummaries.n02Phases,
            n03Summary: parsed.agentSummaries.n03Summary || preset.agentSummaries.n03Summary,
            n04Summary: parsed.agentSummaries.n04Summary || preset.agentSummaries.n04Summary,
          }
        : preset.agentSummaries,
      novaIntro: {
        ...preset.novaIntro,
        text:
          typeof parsed.novaIntroText === 'string' && parsed.novaIntroText.length > 12
            ? personalizeIntroForLearner(parsed.novaIntroText, learnerName)
            : personalizeIntroForLearner(preset.novaIntro.text, learnerName),
      },
    };

    return res.json({
      source: 'gemini',
      analysis: merged,
    });
  } catch (err) {
    console.error('Gemini analyze error, returning deterministic fallback:', err);
    return res.json({
      source: 'fallback',
      analysis: buildFallbackAnalysis(),
    });
  }
});

interface NormalizedSkillInput {
  name: string;
  score: number;
}

interface NormalizedGapInput {
  id: string;
  title: string;
  urgency: 'Urgent' | 'High Gap' | 'Medium Gap';
  impactPercent: number;
  description: string;
}

interface NormalizedPhaseInput {
  phaseNumber: string;
  title: string;
  domainTag: string;
  status: string;
  estimatedHours: number;
  mastery: number;
  summary: string;
}

function inferRequestedIntent(rawIntent: unknown, message: string): NovaIntent {
  if (
    rawIntent === 'explain_roadmap' ||
    rawIntent === 'gap_analysis' ||
    rawIntent === 'mini_project' ||
    rawIntent === 'chat'
  ) {
    if (rawIntent !== 'chat') return rawIntent;
  }
  const lower = message.toLowerCase();
  if (
    lower.includes('mini-project') ||
    lower.includes('mini project') ||
    lower.includes('capstone') ||
    lower.includes('project idea')
  ) {
    return 'mini_project';
  }
  if (
    lower.includes('gap analysis') ||
    lower.includes('missing skill') ||
    lower.includes('priority card') ||
    lower.includes('rank my gaps')
  ) {
    return 'gap_analysis';
  }
  if (
    lower.includes('explain roadmap') ||
    lower.includes('explain my') ||
    lower.includes('phase-by-phase') ||
    lower.includes('phase by phase') ||
    lower.includes('breakdown')
  ) {
    return 'explain_roadmap';
  }
  return 'chat';
}

function inferSkillTopicFromMessage(
  message: string,
  gaps: NormalizedGapInput[],
  activePhase: string
): string {
  const lower = message.toLowerCase();
  const knownTopics = [
    'javascript',
    'typescript',
    'react',
    'node',
    'express',
    'sql',
    'python',
    'pytorch',
    'transformers',
    'rag',
    'kubernetes',
    'docker',
    'terraform',
    'dbt',
    'a/b testing',
    'html',
    'css',
  ];
  for (const topic of knownTopics) {
    if (lower.includes(topic)) {
      if (topic === 'javascript') return 'JavaScript (ES6+)';
      if (topic === 'typescript') return 'TypeScript';
      if (topic === 'react') return 'React';
      if (topic === 'node') return 'Node.js';
      if (topic === 'express') return 'Express APIs';
      if (topic === 'sql') return 'SQL';
      if (topic === 'python') return 'Python';
      if (topic === 'pytorch') return 'PyTorch';
      if (topic === 'transformers') return 'Transformers';
      if (topic === 'rag') return 'RAG & Vector Search';
      if (topic === 'kubernetes') return 'Kubernetes';
      if (topic === 'docker') return 'Docker';
      if (topic === 'terraform') return 'Terraform';
      if (topic === 'dbt') return 'dbt Modeling';
      if (topic === 'a/b testing') return 'A/B Testing';
      if (topic === 'html') return 'Semantic HTML & DOM';
      if (topic === 'css') return 'CSS Layout';
    }
  }
  return gaps[0]?.title || activePhase || 'Core Fundamentals';
}

function detectDeterministicMutation(
  message: string,
  gaps: NormalizedGapInput[],
  activePhase: string
): RoadmapMutationAction {
  const lower = message.toLowerCase();

  if (
    lower.includes('struggl') ||
    lower.includes('stuck') ||
    lower.includes('confus') ||
    lower.includes('hard time') ||
    lower.includes('dont understand') ||
    lower.includes("don't understand") ||
    lower.includes('too hard') ||
    lower.includes('remedial') ||
    lower.includes('help me understand')
  ) {
    const topic = inferSkillTopicFromMessage(message, gaps, activePhase);
    return {
      type: 'inject_remedial',
      topic,
      remedialTitle: `Phase 00: ${topic} Remedial Crash Course`,
      remedialTasks: [
        `Interactive mental-model walkthrough for ${topic}`,
        `3 guided debugging & syntax drills in ${topic}`,
        `Mini confidence check before resuming ${activePhase || 'Phase 01'}`,
      ],
      telemetrySummary: `Adaptive Mentor #N-04 detected friction in ${topic} and injected Phase 00 Remedial Crash Course.`,
      changeLabel: `Injected Phase 00: ${topic} Remedial Crash Course`,
    };
  }

  if (
    lower.includes('faster') ||
    lower.includes('speed up') ||
    lower.includes('accelerate') ||
    lower.includes('quicker') ||
    lower.includes('shorter') ||
    lower.includes('compress') ||
    lower.includes('intensive') ||
    lower.includes('sprint mode')
  ) {
    return {
      type: 'accelerate_plan',
      telemetrySummary:
        'Curriculum Agent #N-02 compressed remaining roadmap phases by ~35% and boosted weekly velocity (+2.5h/wk).',
      changeLabel: 'Compressed roadmap hours (-35%) & boosted velocity (+2.5h/wk)',
    };
  }

  if (
    lower.includes('slower') ||
    lower.includes('slow down') ||
    lower.includes('more time') ||
    lower.includes('more practice') ||
    lower.includes('deeper') ||
    lower.includes('less intense')
  ) {
    return {
      type: 'expand_plan',
      topic: inferSkillTopicFromMessage(message, gaps, activePhase),
      telemetrySummary:
        'Curriculum Agent #N-02 added +6h deep-practice lab time and eased weekly study pacing.',
      changeLabel: 'Added +6h deep-practice drill & eased weekly velocity',
    };
  }

  if (
    lower.includes('focus on') ||
    lower.includes('prioritize') ||
    lower.includes('switch focus') ||
    lower.includes('move to')
  ) {
    const topic = inferSkillTopicFromMessage(message, gaps, activePhase);
    return {
      type: 'focus_topic',
      topic,
      telemetrySummary: `Curriculum Agent #N-02 re-prioritized active study queue around ${topic}.`,
      changeLabel: `Focused active study queue on ${topic}`,
    };
  }

  return { type: 'none' };
}

function buildFallbackBreakdown(
  phases: NormalizedPhaseInput[],
  gaps: NormalizedGapInput[],
  targetRole: string
): NovaRoadmapBreakdownItem[] {
  return phases.map((p, idx) => {
    const matchingGap = gaps[idx] || gaps[0];
    const focusSkill = matchingGap?.title || p.domainTag || p.title;
    return {
      phaseNumber: p.phaseNumber,
      title: p.title,
      missingSkillFocus: focusSkill,
      whyItMatters:
        idx === 0
          ? `Closes your #1 blocker (${focusSkill}) required by ${targetRole} hiring rubrics before unlocking higher-leverage modules.`
          : `Builds production depth in ${p.domainTag || focusSkill} (${p.summary || 'core workflows'}) needed to operate end-to-end as a ${targetRole}.`,
      estimatedHours: p.estimatedHours || 20,
    };
  });
}

function buildFallbackGapPriority(
  gaps: NormalizedGapInput[],
  phases: NormalizedPhaseInput[],
  targetRole: string
): NovaGapPriorityItem[] {
  const sorted = [...gaps].sort(
    (a, b) => Math.abs(b.impactPercent) - Math.abs(a.impactPercent)
  );
  return sorted.map((gap, idx) => {
    const matchedPhase = phases.find((p) => {
      const target = `${p.title} ${p.domainTag} ${p.summary}`.toLowerCase();
      const firstWord = gap.title.toLowerCase().split(/[\s(/]+/)[0];
      return firstWord && target.includes(firstWord);
    });
    const estimatedHours = matchedPhase?.estimatedHours || (idx === 0 ? 24 : idx === 1 ? 18 : 14);
    return {
      id: gap.id,
      rank: idx + 1,
      title: gap.title,
      urgency: gap.urgency,
      impactPercent: Math.abs(gap.impactPercent) || (idx === 0 ? 40 : 28),
      estimatedHours,
      reason:
        gap.description ||
        `Core technical requirement for ${targetRole} interviews and production delivery.`,
    };
  });
}

function buildFallbackMiniProject(
  targetRole: string,
  skills: NormalizedSkillInput[],
  gaps: NormalizedGapInput[]
): NovaMiniProjectIdea {
  const existing = skills.slice(0, 2).map((s) => s.name);
  const existingLabel = existing.length > 0 ? existing : ['Python', 'SQL'];
  const missing = gaps.slice(0, 2).map((g) => g.title);
  const missingLabel = missing.length > 0 ? missing : ['Core Target Stack'];

  if (targetRole === 'Full Stack Developer') {
    return {
      id: `capstone-${Date.now()}`,
      title: 'Live SQL Telemetry & Job Queue Dashboard',
      summary: `Bridge your ${existingLabel.join(' + ')} strengths with ${missingLabel.join(' & ')} by building an interactive full-stack analytics console backed by SQL queries and a Python ingestion script.`,
      existingSkillsUsed: existingLabel,
      missingSkillsTargeted: missingLabel,
      estimatedHours: 14,
      tasks: [
        `Write a ${existingLabel[0] || 'Python'} + ${existingLabel[1] || 'SQL'} seed & query script exporting metrics JSON`,
        `Build an interactive ${missingLabel[0] || 'ES6+ / React'} dashboard view with live filtering & state`,
        `Add a REST API endpoint & verify end-to-end data roundtrip`,
      ],
    };
  }

  if (targetRole === 'AI / ML Engineer') {
    return {
      id: `capstone-${Date.now()}`,
      title: 'Hybrid SQL-Backed RAG Evaluation Harness',
      summary: `Combine ${existingLabel.join(' + ')} with ${missingLabel.join(' & ')} to index technical docs, run vector retrieval, and log latency/precision scores in SQL.`,
      existingSkillsUsed: existingLabel,
      missingSkillsTargeted: missingLabel,
      estimatedHours: 16,
      tasks: [
        `Chunk & embed a domain corpus using ${existingLabel[0] || 'Python'}`,
        `Implement ${missingLabel[0] || 'PyTorch / Vector Retrieval'} top-k ranking pipeline`,
        `Log precision@k & latency metrics to ${existingLabel[1] || 'SQL'}`,
      ],
    };
  }

  if (targetRole === 'Data Analyst') {
    return {
      id: `capstone-${Date.now()}`,
      title: 'Product Funnel & Experimentation Audit Pack',
      summary: `Pair your ${existingLabel.join(' + ')} foundation with ${missingLabel.join(' & ')} to model retention cohorts and evaluate an A/B experiment end-to-end.`,
      existingSkillsUsed: existingLabel,
      missingSkillsTargeted: missingLabel,
      estimatedHours: 12,
      tasks: [
        `Clean event logs & compute cohort retention in ${existingLabel[1] || 'SQL'}`,
        `Run significance & lift tests for ${missingLabel[0] || 'A/B Testing'}`,
        `Publish an executive KPI scorecard targeting ${missingLabel[1] || 'BI & Visualization'}`,
      ],
    };
  }

  return {
    id: `capstone-${Date.now()}`,
    title: 'Containerized Telemetry & GitOps Deploy Pipeline',
    summary: `Unite your ${existingLabel.join(' + ')} automation skills with ${missingLabel.join(' & ')} by packaging a monitored service with reproducible IaC and health probes.`,
    existingSkillsUsed: existingLabel,
    missingSkillsTargeted: missingLabel,
    estimatedHours: 14,
    tasks: [
      `Author a ${existingLabel[0] || 'Python'} health-check CLI & database probe`,
      `Containerize and deploy with ${missingLabel[0] || 'Docker & Kubernetes'} manifests`,
      `Instrument live metrics & alert rules for ${missingLabel[1] || 'Observability'}`,
    ],
  };
}

function buildStructuredNovaFallback(params: {
  intent: NovaIntent;
  message: string;
  learnerName: string;
  targetRole: string;
  readinessScore: number;
  skills: NormalizedSkillInput[];
  gaps: NormalizedGapInput[];
  phases: NormalizedPhaseInput[];
  activePhase: string;
  hasRemedial: boolean;
}) {
  const firstName = params.learnerName.split(/\s+/)[0] || params.learnerName;
  const skillSummary =
    params.skills.map((s) => `${s.name} (${s.score}%)`).join(', ') || 'Baseline skills';
  const topGap = params.gaps[0]?.title || params.activePhase || 'Core Target Stack';

  if (params.intent === 'explain_roadmap') {
    const breakdown = buildFallbackBreakdown(params.phases, params.gaps, params.targetRole);
    return {
      reply: `Here is your personalized ${params.targetRole} phase-by-phase breakdown, ${firstName}. Each stage targets one high-leverage missing capability while building on your ${skillSummary} foundation.`,
      roadmapBreakdown: breakdown,
      mutation: { type: 'none' } as RoadmapMutationAction,
    };
  }

  if (params.intent === 'gap_analysis') {
    const gapPriorityCard = buildFallbackGapPriority(params.gaps, params.phases, params.targetRole);
    return {
      reply:
        gapPriorityCard.length > 0
          ? `Ranked your ${gapPriorityCard.length} remaining gaps for ${params.targetRole} by job-market impact and estimated study time. Clearing #1 (${gapPriorityCard[0].title}) yields the fastest lift from ${params.readinessScore}% readiness.`
          : `Zero critical skill gaps remain for ${params.targetRole} right now! Keep completing phase milestones to push your readiness toward 100%.`,
      gapPriorityCard,
      mutation: { type: 'none' } as RoadmapMutationAction,
    };
  }

  if (params.intent === 'mini_project') {
    const projectIdea = buildFallbackMiniProject(params.targetRole, params.skills, params.gaps);
    return {
      reply: `Built a custom capstone mini-project for you, ${firstName}: "${projectIdea.title}". It pairs your verified ${projectIdea.existingSkillsUsed.join(' + ')} strengths directly with ${projectIdea.missingSkillsTargeted.join(' & ')} — click "+ Add Project to Roadmap" below to schedule it.`,
      projectIdea,
      mutation: { type: 'none' } as RoadmapMutationAction,
    };
  }

  const mutation = detectDeterministicMutation(params.message, params.gaps, params.activePhase);
  if (mutation.type === 'inject_remedial') {
    return {
      reply: `No worries, ${firstName} — ${mutation.topic} trips up lots of learners transitioning into ${params.targetRole}. I just injected "${mutation.remedialTitle}" at the top of your Roadmap and logged an Adaptive Mentor (#N-04) intervention so you can build rock-solid intuition before your next check.`,
      mutation,
    };
  }
  if (mutation.type === 'accelerate_plan') {
    return {
      reply: `Accelerated mode activated, ${firstName}! I compressed your remaining ${params.targetRole} phase hours by ~35%, bumped your target velocity by +2.5h/wk, and logged the curriculum re-sequencing with Agent #N-02.`,
      mutation,
    };
  }
  if (mutation.type === 'expand_plan') {
    return {
      reply: `Adjusted your pace, ${firstName}. I added +6h of hands-on practice drills to your active phase and eased your weekly velocity so you can master ${mutation.topic || params.activePhase} without rushing.`,
      mutation,
    };
  }
  if (mutation.type === 'focus_topic') {
    return {
      reply: `Prioritized ${mutation.topic} on your ${params.targetRole} Roadmap and logged the focus update with Agent #N-02.`,
      mutation,
    };
  }

  return {
    reply: `At ${params.readinessScore}% readiness for ${params.targetRole}, your highest-leverage step is "${params.activePhase}" to close ${topGap}${params.hasRemedial ? ' (clear Phase 00 Remedial first so your next quiz passes cleanly)' : ''}. You can also ask me to adjust your plan pace, generate a mini-project, or explain any phase.`,
    mutation: { type: 'none' } as RoadmapMutationAction,
  };
}

app.post('/api/nova/chat', async (req, res) => {
  const {
    message = '',
    intent: rawIntent = 'chat',
    learnerName = 'Arjun K.',
    targetRole = 'Full Stack Developer',
    readinessScore = 34,
    skills = [],
    gaps = [],
    phases = [],
    activePhase = '',
    hasRemedial = false,
  } = req.body ?? {};

  const trimmed = String(message).trim();
  if (!trimmed) {
    return res.status(400).json({ error: 'Message is required' });
  }

  const intent = inferRequestedIntent(rawIntent, trimmed);

  const normalizedSkills: NormalizedSkillInput[] = Array.isArray(skills)
    ? skills
        .filter((s: any) => s && typeof s.name === 'string')
        .map((s: any) => ({ name: String(s.name), score: Number(s.score) || 70 }))
    : [];

  const normalizedGaps: NormalizedGapInput[] = Array.isArray(gaps)
    ? gaps
        .filter((g: any) => g && typeof g.title === 'string')
        .map((g: any, idx: number) => ({
          id: String(g.id || `gap-${idx}`),
          title: String(g.title),
          urgency:
            g.urgency === 'Urgent' || g.urgency === 'High Gap' || g.urgency === 'Medium Gap'
              ? g.urgency
              : idx === 0
                ? 'Urgent'
                : 'High Gap',
          impactPercent: typeof g.impactPercent === 'number' ? g.impactPercent : -30,
          description: String(g.description || ''),
        }))
    : [];

  const normalizedPhases: NormalizedPhaseInput[] = Array.isArray(phases)
    ? phases
        .filter((p: any) => p && typeof p.title === 'string')
        .map((p: any, idx: number) => ({
          phaseNumber: String(p.phaseNumber || `Phase 0${idx + 1}`),
          title: String(p.title),
          domainTag: String(p.domainTag || 'Core'),
          status: String(p.status || 'not-started'),
          estimatedHours: Number(p.estimatedHours) || 20,
          mastery: Number(p.mastery) || 0,
          summary: String(p.summary || ''),
        }))
    : [];

  const fallbackPayload = buildStructuredNovaFallback({
    intent,
    message: trimmed,
    learnerName,
    targetRole,
    readinessScore,
    skills: normalizedSkills,
    gaps: normalizedGaps,
    phases: normalizedPhases,
    activePhase,
    hasRemedial,
  });

  const ai = getGeminiClient();
  if (!ai) {
    return res.json({
      source: 'fallback',
      intent,
      ...fallbackPayload,
    });
  }

  const skillSummary =
    normalizedSkills.map((s) => `${s.name} (${s.score}%)`).join(', ') || 'None entered yet';
  const gapSummary =
    normalizedGaps
      .map((g) => `${g.title} [${g.urgency}, impact ${g.impactPercent}%]: ${g.description}`)
      .join(' | ') || 'No active gaps';
  const phaseSummary =
    normalizedPhases
      .map((p) => `${p.phaseNumber}: ${p.title} (${p.domainTag}, ${p.estimatedHours}h, ${p.status})`)
      .join(' | ') || activePhase;

  try {
    const response = await ai.models.generateContent({
      model: 'gemini-4-hydrogen',
      contents: `Intent: ${intent}
Learner message: "${trimmed}"`,
      config: {
        thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
        responseMimeType: 'application/json',
        systemInstruction: `You are Nova, PathForge AI's Adaptive Mentor Agent (#N-04) coaching ${learnerName}.
Target Role: ${targetRole}
Current Readiness: ${readinessScore}%
Existing Skills: ${skillSummary}
Missing Skill Gaps: ${gapSummary}
Roadmap Phases: ${phaseSummary}
Active Phase: ${activePhase || 'Phase 01'}
Remedial Active: ${hasRemedial ? 'Yes' : 'No'}

Rules by intent:
1. If intent is "explain_roadmap": Provide a crisp 2-sentence reply AND populate "roadmapBreakdown" with one entry per non-remedial roadmap phase explaining what missing skill it targets and specifically why that skill matters for a ${targetRole}. Set mutation.type="none".
2. If intent is "gap_analysis": Provide a crisp 2-sentence reply AND populate "gapPriorityCard" ranking the Missing Skill Gaps from highest market impact to lowest, including estimatedHours to master and a concise 1-sentence reason. Set mutation.type="none".
3. If intent is "mini_project": Provide a crisp 2-sentence reply AND populate "projectIdea" with ONE tailored hands-on capstone project combining ${learnerName}'s Existing Skills with their Missing Skill Gaps (include existingSkillsUsed, missingSkillsTargeted, estimatedHours 8-18, and 3 concrete build tasks). Set mutation.type="none".
4. If intent is "chat": Reply helpfully in 2-4 crisp sentences. If the learner says they are struggling/stuck/confused on a skill or asks to modify/speed up/slow down/prioritize their roadmap, ALSO set "mutation.type" to one of:
   - "inject_remedial" (when struggling/stuck/confused — include topic, remedialTitle like "Phase 00: <Topic> Remedial Crash Course", 3 remedialTasks, telemetrySummary, changeLabel)
   - "accelerate_plan" (when asking to make plan faster/shorter/compress — include telemetrySummary, changeLabel)
   - "expand_plan" (when asking to slow down/add more practice — include topic, telemetrySummary, changeLabel)
   - "focus_topic" (when asking to focus/prioritize a specific skill — include topic, telemetrySummary, changeLabel)
   Otherwise set mutation.type="none".`,
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            reply: { type: Type.STRING },
            roadmapBreakdown: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  phaseNumber: { type: Type.STRING },
                  title: { type: Type.STRING },
                  missingSkillFocus: { type: Type.STRING },
                  whyItMatters: { type: Type.STRING },
                  estimatedHours: { type: Type.INTEGER },
                },
                required: [
                  'phaseNumber',
                  'title',
                  'missingSkillFocus',
                  'whyItMatters',
                  'estimatedHours',
                ],
              },
            },
            gapPriorityCard: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  id: { type: Type.STRING },
                  rank: { type: Type.INTEGER },
                  title: { type: Type.STRING },
                  urgency: { type: Type.STRING },
                  impactPercent: { type: Type.INTEGER },
                  estimatedHours: { type: Type.INTEGER },
                  reason: { type: Type.STRING },
                },
                required: [
                  'id',
                  'rank',
                  'title',
                  'urgency',
                  'impactPercent',
                  'estimatedHours',
                  'reason',
                ],
              },
            },
            projectIdea: {
              type: Type.OBJECT,
              properties: {
                id: { type: Type.STRING },
                title: { type: Type.STRING },
                summary: { type: Type.STRING },
                existingSkillsUsed: { type: Type.ARRAY, items: { type: Type.STRING } },
                missingSkillsTargeted: { type: Type.ARRAY, items: { type: Type.STRING } },
                estimatedHours: { type: Type.INTEGER },
                tasks: { type: Type.ARRAY, items: { type: Type.STRING } },
              },
              required: [
                'id',
                'title',
                'summary',
                'existingSkillsUsed',
                'missingSkillsTargeted',
                'estimatedHours',
                'tasks',
              ],
            },
            mutation: {
              type: Type.OBJECT,
              properties: {
                type: { type: Type.STRING },
                topic: { type: Type.STRING },
                remedialTitle: { type: Type.STRING },
                remedialTasks: { type: Type.ARRAY, items: { type: Type.STRING } },
                telemetrySummary: { type: Type.STRING },
                changeLabel: { type: Type.STRING },
              },
              required: ['type'],
            },
          },
          required: ['reply'],
        },
      },
    });

    const parsed = JSON.parse((response.text ?? '{}').trim());
    const reply =
      typeof parsed.reply === 'string' && parsed.reply.trim().length > 0
        ? parsed.reply.trim()
        : fallbackPayload.reply;

    const deterministicMutation = detectDeterministicMutation(
      trimmed,
      normalizedGaps,
      activePhase
    );
    const modelMutationType = parsed.mutation?.type;
    const validMutationType =
      modelMutationType === 'inject_remedial' ||
      modelMutationType === 'accelerate_plan' ||
      modelMutationType === 'expand_plan' ||
      modelMutationType === 'focus_topic'
        ? modelMutationType
        : 'none';

    const resolvedMutation: RoadmapMutationAction =
      validMutationType !== 'none'
        ? {
            type: validMutationType,
            topic: parsed.mutation?.topic || deterministicMutation.topic,
            remedialTitle:
              parsed.mutation?.remedialTitle || deterministicMutation.remedialTitle,
            remedialTasks:
              Array.isArray(parsed.mutation?.remedialTasks) &&
              parsed.mutation.remedialTasks.length > 0
                ? parsed.mutation.remedialTasks.slice(0, 4).map(String)
                : deterministicMutation.remedialTasks,
            telemetrySummary:
              parsed.mutation?.telemetrySummary || deterministicMutation.telemetrySummary,
            changeLabel: parsed.mutation?.changeLabel || deterministicMutation.changeLabel,
          }
        : intent === 'chat'
          ? deterministicMutation
          : { type: 'none' };

    const result: Record<string, any> = {
      source: 'gemini',
      intent,
      reply,
      mutation: resolvedMutation,
    };

    if (intent === 'explain_roadmap') {
      result.roadmapBreakdown =
        Array.isArray(parsed.roadmapBreakdown) && parsed.roadmapBreakdown.length > 0
          ? parsed.roadmapBreakdown.map((item: any, idx: number) => ({
              phaseNumber: String(item.phaseNumber || normalizedPhases[idx]?.phaseNumber || `Phase 0${idx + 1}`),
              title: String(item.title || normalizedPhases[idx]?.title || 'Module'),
              missingSkillFocus: String(item.missingSkillFocus || normalizedGaps[idx]?.title || 'Core Skill'),
              whyItMatters: String(item.whyItMatters || ''),
              estimatedHours: Number(item.estimatedHours) || normalizedPhases[idx]?.estimatedHours || 20,
            }))
          : fallbackPayload.roadmapBreakdown;
    }

    if (intent === 'gap_analysis') {
      result.gapPriorityCard =
        Array.isArray(parsed.gapPriorityCard) && parsed.gapPriorityCard.length > 0
          ? parsed.gapPriorityCard
              .map((g: any, idx: number) => ({
                id: String(g.id || normalizedGaps[idx]?.id || `gap-${idx}`),
                rank: Number(g.rank) || idx + 1,
                title: String(g.title || normalizedGaps[idx]?.title || 'Skill Gap'),
                urgency:
                  g.urgency === 'Urgent' || g.urgency === 'High Gap' || g.urgency === 'Medium Gap'
                    ? g.urgency
                    : normalizedGaps[idx]?.urgency || 'High Gap',
                impactPercent: Math.abs(Number(g.impactPercent) || normalizedGaps[idx]?.impactPercent || 30),
                estimatedHours: Math.max(4, Number(g.estimatedHours) || 18),
                reason: String(g.reason || normalizedGaps[idx]?.description || ''),
              }))
              .sort((a: NovaGapPriorityItem, b: NovaGapPriorityItem) => b.impactPercent - a.impactPercent)
              .map((g: NovaGapPriorityItem, idx: number) => ({ ...g, rank: idx + 1 }))
          : fallbackPayload.gapPriorityCard;
    }

    if (intent === 'mini_project') {
      const p = parsed.projectIdea;
      result.projectIdea =
        p && typeof p.title === 'string' && p.title.trim().length > 0
          ? {
              id: `capstone-${Date.now()}`,
              title: p.title.trim(),
              summary: String(p.summary || fallbackPayload.projectIdea?.summary || ''),
              existingSkillsUsed:
                Array.isArray(p.existingSkillsUsed) && p.existingSkillsUsed.length > 0
                  ? p.existingSkillsUsed.slice(0, 3).map(String)
                  : fallbackPayload.projectIdea?.existingSkillsUsed || ['Python', 'SQL'],
              missingSkillsTargeted:
                Array.isArray(p.missingSkillsTargeted) && p.missingSkillsTargeted.length > 0
                  ? p.missingSkillsTargeted.slice(0, 3).map(String)
                  : fallbackPayload.projectIdea?.missingSkillsTargeted || ['Core Stack'],
              estimatedHours: Math.max(6, Math.min(28, Number(p.estimatedHours) || 14)),
              tasks:
                Array.isArray(p.tasks) && p.tasks.length > 0
                  ? p.tasks.slice(0, 4).map(String)
                  : fallbackPayload.projectIdea?.tasks || [],
            }
          : fallbackPayload.projectIdea;
    }

    return res.json(result);
  } catch (err) {
    console.error('Gemini chat error, returning structured fallback:', err);
    return res.json({
      source: 'fallback',
      intent,
      ...fallbackPayload,
    });
  }
});

async function startServer() {
  const PORT = 3000;

  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.resolve(__dirname, 'dist');
    app.use(express.static(distPath));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`PathForge AI server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer();
