import { Fragment, useState } from 'react';
import { comboToClass, buildComboClassMap } from '@/lib/comboUtils';
import { actionColor } from '@/lib/gridModel';

export interface ComboBreakdownPanelProps {
  handClass: string | null;
  combos: string[];
  weights: number[];
  equity: number[];
  strategy?: number[];
  ev_detail?: number[];
  actions?: string[];
  hero_action?: string | null;
  onClose: () => void;
}

export default function ComboBreakdownPanel({
  handClass,
  combos,
  weights,
  equity,
  strategy,
  ev_detail,
  actions,
  hero_action,
  onClose,
}: ComboBreakdownPanelProps) {
  const [expanded, setExpanded] = useState<boolean>(false);

  if (!handClass) return null;

  // Gather indices for this class: try buildComboClassMap first, fall back to filter,
  // then fall back to all combos (parent pre-selected when utilities return no match).
  const classMap = buildComboClassMap(combos);
  const classData = classMap.classToData[handClass];
  let classIndices: number[];
  if (classData?.indices && classData.indices.length > 0) {
    classIndices = classData.indices;
  } else {
    const filtered = combos
      .map((c, i) => (comboToClass(c) === handClass ? i : -1))
      .filter((i) => i >= 0);
    classIndices = filtered.length > 0 ? filtered : combos.map((_, i) => i);
  }

  const nCombos = combos.length;
  const isHeroSeat = !!(strategy && ev_detail && actions && actions.length > 0);
  const heroActionIdx = isHeroSeat ? actions!.indexOf(hero_action ?? '') : -1;

  // Aggregate L1 action mix + EV per action over the class.
  const totalClassWeight = classIndices.reduce((s, h) => s + weights[h], 0);

  interface ActionAgg {
    freq: number;
    ev: number;
  }

  const actionAggs: ActionAgg[] = isHeroSeat
    ? actions!.map((_, a) => {
        const freq =
          totalClassWeight > 0
            ? classIndices.reduce((s, h) => s + weights[h] * strategy![a * nCombos + h], 0) /
              totalClassWeight
            : 0;
        const ev =
          totalClassWeight > 0
            ? classIndices.reduce((s, h) => s + weights[h] * ev_detail![a * nCombos + h], 0) /
              totalClassWeight
            : 0;
        return { freq, ev };
      })
    : [];

  const classEquity =
    totalClassWeight > 0
      ? classIndices.reduce((s, h) => s + weights[h] * equity[h], 0) / totalClassWeight
      : 0;

  const maxEv = isHeroSeat ? Math.max(...actionAggs.map((a) => a.ev)) : 0;
  const heroEv = isHeroSeat && heroActionIdx >= 0 ? actionAggs[heroActionIdx].ev : maxEv;
  const evLoss = isHeroSeat ? Math.max(0, maxEv - heroEv) : 0;

  return (
    <div className="section">
      <div className="section-head">
        <div className="section-head-l">
          <span className="section-head-title">COMBO BREAKDOWN</span>
          <span className="section-head-meta">· {handClass}</span>
        </div>
        <div className="section-head-r">
          <button
            className="btn btn-small btn-ghost"
            onClick={onClose}
            type="button"
            aria-label="close"
          >
            ✕
          </button>
        </div>
      </div>
      <div className="section-body">
        <table>
          <tbody>
            <tr
              className="l1-row"
              data-testid="combo-breakdown-l1"
              onClick={() => setExpanded((e) => !e)}
              style={{ cursor: 'pointer' }}
            >
              <td colSpan={4}>
                {isHeroSeat ? (
                  <>
                    <div className="cap" style={{ marginBottom: 4 }}>
                      action mix · ev-loss: {evLoss.toFixed(3)} chips
                    </div>
                    {actions!.map((action, a) => {
                      const isHeroAction = action === hero_action;
                      return (
                        <div
                          key={action}
                          className={'dist-row' + (isHeroAction ? ' active' : '')}
                          aria-label={isHeroAction ? 'hero-action' : undefined}
                        >
                          <span className="dist-row-act">
                            <span style={{ display: 'inline-block', width: 8, height: 8, background: actionColor(action), marginRight: 6, verticalAlign: 'middle' }} />
                            {isHeroAction ? '★ ' : ''}{action}
                          </span>
                          <span className="dist-row-freq">
                            {(actionAggs[a].freq * 100).toFixed(1)}%
                          </span>
                          <div className="dist-row-bar">
                            <div
                              className="dist-row-bar-fill"
                              style={{ width: `${actionAggs[a].freq * 100}%`, background: actionColor(action) }}
                            />
                          </div>
                          <span className="dist-row-delta mono-num">
                            {actionAggs[a].ev.toFixed(3)}
                          </span>
                        </div>
                      );
                    })}
                  </>
                ) : (
                  <div className="dist-row">
                    <span className="dist-row-act">equity</span>
                    <span className="dist-row-freq">{(classEquity * 100).toFixed(1)}%</span>
                    <span className="dist-row-delta muted">
                      weight: {totalClassWeight.toFixed(2)}
                    </span>
                  </div>
                )}
              </td>
            </tr>

            {expanded &&
              classIndices.map((h) => {
                const comboWeight = weights[h];
                const comboFreq =
                  totalClassWeight > 0 ? comboWeight / totalClassWeight : 0;
                const comboEv =
                  isHeroSeat && heroActionIdx >= 0
                    ? ev_detail![heroActionIdx * nCombos + h]
                    : null;
                const comboEquity = equity[h];

                return (
                  <Fragment key={combos[h]}>
                    <tr
                      className="l2-row"
                      data-testid="combo-breakdown-l2-row"
                    >
                      <td>{combos[h]}</td>
                      <td className="mono-num">{(comboFreq * 100).toFixed(1)}%</td>
                      <td className="mono-num">
                        {comboEv != null ? comboEv.toFixed(3) : '—'}
                      </td>
                      <td className="mono-num">{(comboEquity * 100).toFixed(1)}%</td>
                    </tr>
                  </Fragment>
                );
              })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
