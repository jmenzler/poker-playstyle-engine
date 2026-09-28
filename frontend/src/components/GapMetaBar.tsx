import { truncateClusterKey } from '@/lib/format';

interface GapMetaBarProps {
  cluster_key: string;
  max_neighbor_distance: number;
  n_players_active: number;
  street: string;
  is_multiway: boolean;
  priorGtoScore?: number | null;
}

export default function GapMetaBar({
  cluster_key,
  max_neighbor_distance,
  n_players_active,
  street,
  is_multiway,
  priorGtoScore,
}: GapMetaBarProps) {
  return (
    <div className="text-[12px] text-fg-muted flex gap-4 flex-wrap">
      <span>
        cluster: <span className="text-fg-cluster">{truncateClusterKey(cluster_key)}</span>
      </span>
      <span>distance: {max_neighbor_distance.toFixed(3)}</span>
      <span>players: {n_players_active}</span>
      <span>street: {street}</span>
      {is_multiway && <span className="text-warning">[MULTIWAY]</span>}
      {priorGtoScore != null && <span>prior gto_score: {priorGtoScore}</span>}
    </div>
  );
}
