import { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Swords, Shield, Skull, ChevronDown, ChevronUp, ScrollText } from 'lucide-react';

// ---------------------------------------------------------------------------
// Frozen UI props contract (Spec §9, src/dnd_auto_dmg/state.py):
//
//   props.langgraphState = {
//     combatants: { "<id>": { ...Combatant.model_dump()... } },
//     round: 1,
//     log: ["last", "10", "event_log", "lines"]
//   }
//
// Combatant.model_dump() shape:
//   { id, name, kind: "pc"|"npc"|"monster", max_hp, hp, ac,
//     attributes: {}, features: [], statuses: [{name, source,
//     duration_rounds, applied_round, notes}], inventory: [],
//     origin: "roster"|"adhoc", is_alive }
//
// This component reads ONLY these fields. Everything is accessed
// defensively so a missing/undefined langgraphState (before the first
// turn) renders a placeholder instead of throwing.
// ---------------------------------------------------------------------------

const KIND_ORDER = { pc: 0, npc: 1, monster: 2 };

const KIND_LABEL = {
  pc: 'PC',
  npc: 'NPC',
  monster: 'Monster',
};

const KIND_BADGE_CLASS = {
  pc: 'bg-blue-600 text-white hover:bg-blue-600',
  npc: 'bg-purple-600 text-white hover:bg-purple-600',
  monster: 'bg-orange-600 text-white hover:bg-orange-600',
};

function prettifyStatusName(name) {
  if (!name || typeof name !== 'string') return 'unknown status';
  return name.split('_').join(' ');
}

function hpBarColorClass(pct, isAlive) {
  if (!isAlive) return 'bg-gray-400';
  if (pct > 50) return 'bg-green-500';
  if (pct >= 20) return 'bg-amber-500';
  return 'bg-red-500';
}

function sortCombatants(combatants) {
  return Object.values(combatants || {}).sort((a, b) => {
    const orderA = KIND_ORDER[a?.kind] !== undefined ? KIND_ORDER[a.kind] : 3;
    const orderB = KIND_ORDER[b?.kind] !== undefined ? KIND_ORDER[b.kind] : 3;
    if (orderA !== orderB) return orderA - orderB;
    const nameA = a?.name || '';
    const nameB = b?.name || '';
    return nameA.localeCompare(nameB);
  });
}

function CombatantCard({ combatant }) {
  const {
    id,
    name,
    kind,
    max_hp: maxHp,
    hp,
    ac,
    statuses,
    origin,
    is_alive: isAlive,
  } = combatant || {};

  const safeMaxHp = typeof maxHp === 'number' ? maxHp : 0;
  const safeHp = typeof hp === 'number' ? hp : 0;
  const rawPct = safeMaxHp > 0 ? (safeHp / safeMaxHp) * 100 : 0;
  const pct = Math.max(0, Math.min(100, rawPct));
  const alive = isAlive !== false;
  const safeStatuses = Array.isArray(statuses) ? statuses : [];

  return (
    <Card
      key={id}
      className={`p-3 space-y-2 ${!alive ? 'opacity-75 border-gray-300' : ''}`}
    >
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5 min-w-0">
          {!alive && <Skull className="w-4 h-4 text-gray-500 shrink-0" />}
          <span className="font-medium truncate">{name || id || 'Unknown'}</span>
          <Badge className={`text-xs shrink-0 ${KIND_BADGE_CLASS[kind] || 'bg-gray-500 text-white hover:bg-gray-500'}`}>
            {KIND_LABEL[kind] || kind || '?'}
          </Badge>
          {origin === 'adhoc' && (
            <span className="text-[10px] text-gray-500 italic shrink-0">ad-hoc</span>
          )}
        </div>
        {(ac !== null && ac !== undefined) && (
          <div className="flex items-center gap-1 text-xs text-gray-600 shrink-0">
            <Shield className="w-3.5 h-3.5" />
            <span>{ac}</span>
          </div>
        )}
      </div>

      <div className="space-y-1">
        <div className="w-full h-2.5 rounded-full bg-gray-200 overflow-hidden">
          <div
            className={`h-full rounded-full transition-all ${hpBarColorClass(pct, alive)}`}
            style={{ width: `${pct}%` }}
          />
        </div>
        <div className="text-xs text-gray-600">
          {safeHp} / {safeMaxHp} HP
          {!alive && <span className="ml-1 text-gray-500">(down)</span>}
        </div>
      </div>

      {safeStatuses.length > 0 && (
        <div className="flex flex-wrap gap-1 pt-1">
          {safeStatuses.map((status, idx) => {
            const statusName = status?.name;
            const isDireStatus = statusName === 'dead' || statusName === 'unconscious';
            const duration = status?.duration_rounds;
            const label = `${prettifyStatusName(statusName)}${
              duration !== null && duration !== undefined ? ` (${duration})` : ''
            }`;
            return (
              <Badge
                key={`${statusName}-${idx}`}
                variant="outline"
                className={`text-[10px] ${
                  isDireStatus ? 'border-red-400 text-red-600' : 'border-gray-300 text-gray-600'
                }`}
              >
                {label}
              </Badge>
            );
          })}
        </div>
      )}
    </Card>
  );
}

export default function LanggraphStateDisplay() {
  const [logExpanded, setLogExpanded] = useState(false);

  const langgraphState =
    (typeof props !== 'undefined' && props && props.langgraphState) || undefined;

  const combatants = langgraphState?.combatants || {};
  const round = langgraphState?.round;
  const log = Array.isArray(langgraphState?.log) ? langgraphState.log : [];

  const sortedCombatants = sortCombatants(combatants);
  const hasCombatants = sortedCombatants.length > 0;

  return (
    <Card className="w-full mb-4 p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Swords className="w-5 h-5 text-slate-700" />
          <h3 className="text-lg font-semibold text-slate-900">Combat Tracker</h3>
        </div>
        {round !== null && round !== undefined && (
          <Badge className="bg-slate-800 text-white hover:bg-slate-800 text-sm">
            Round {round}
          </Badge>
        )}
      </div>

      <Separator className="mb-3" />

      {!hasCombatants && (
        <div className="text-sm text-gray-500 italic py-4 text-center">
          No combatants yet — describe an attack to begin.
        </div>
      )}

      {hasCombatants && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {sortedCombatants.map((combatant) => (
            <CombatantCard key={combatant?.id} combatant={combatant} />
          ))}
        </div>
      )}

      <Separator className="my-3" />

      <div>
        <button
          type="button"
          onClick={() => setLogExpanded((prev) => !prev)}
          className="flex items-center gap-2 text-sm font-medium text-slate-700 hover:text-slate-900 transition-colors w-full"
        >
          <ScrollText className="w-4 h-4" />
          <span>Event Log</span>
          <span className="text-xs text-gray-500 font-normal">({log.length})</span>
          <span className="ml-auto">
            {logExpanded ? (
              <ChevronUp className="w-4 h-4" />
            ) : (
              <ChevronDown className="w-4 h-4" />
            )}
          </span>
        </button>

        {logExpanded && (
          <div className="mt-2 p-2 bg-gray-50 rounded-md border border-gray-200 max-h-56 overflow-y-auto">
            {log.length === 0 ? (
              <div className="text-xs text-gray-500 italic">No events yet.</div>
            ) : (
              <ol className="space-y-1">
                {log.map((line, idx) => (
                  <li key={idx} className="text-xs text-gray-700">
                    <span className="text-gray-400 mr-1">{idx + 1}.</span>
                    {line}
                  </li>
                ))}
              </ol>
            )}
          </div>
        )}
      </div>
    </Card>
  );
}
