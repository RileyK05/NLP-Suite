import type { Table } from "./api";
import type { Settings as ChartSettings } from "./chartLayout";
import type { ChartSeed } from "./VisualizeDialog";
import { publicationGaps, type ChartContract } from "./views";
import { useChartSettings, Workbench } from "./Workbench";

export function RunWorkbench({
  table,
  artifactPath,
  contract,
  canPublish,
  onPublish,
  onHandOff,
}: {
  table: Table;
  artifactPath: string;
  contract: ChartContract;
  canPublish: boolean;
  onPublish: (settings: ChartSettings) => void;
  onHandOff?: (seed: ChartSeed) => void;
}) {
  const { settings, setSettings, selected, setSelected } =
    useChartSettings(table);
  return (
    <Workbench
      table={table}
      artifactPath={artifactPath}
      settings={settings}
      onSettings={setSettings}
      selected={selected}
      onSelect={setSelected}
      canPublish={canPublish}
      onPublish={onPublish}
      gaps={publicationGaps(contract, settings)}
      onHandOff={onHandOff}
    />
  );
}
