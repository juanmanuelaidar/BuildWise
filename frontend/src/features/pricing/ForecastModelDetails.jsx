import { Alert, Box, Card, CardContent, Chip, Stack, Typography, Accordion, AccordionDetails, AccordionSummary } from "@mui/material";
import ExpandMoreIconModule from "@mui/icons-material/ExpandMore";

import { resolveMuiIcon } from "../../shared/components/resolveMuiIcon.js";
import { formatNumber } from "../../shared/utils/formatters.js";
import { getModelDisplayName, getRegressorDisplayName } from "./forecastModelLabels.js";

const ExpandMoreIcon = resolveMuiIcon(ExpandMoreIconModule);

function confidenceColor(value) {
  const normalized = String(value || "").toLowerCase();
  if (normalized === "alta") return "success";
  if (normalized === "media") return "warning";
  if (normalized === "baja") return "error";
  return "default";
}

export function ForecastModelDetails({ selection, title = "Detalles del modelo", compact = false }) {
  if (!selection) return null;

  return (
    <Card className={compact ? "mt-5" : "my-4"}>
      <CardContent className="p-4">
        <Stack spacing={compact ? 1.5 : 2}>
          <Box>
            <Typography variant="overline" color="text.secondary">
              {title}
            </Typography>
            <Typography variant={compact ? "h4" : "h3"} mt={0.5}>
              {getModelDisplayName(selection.modelo_resuelto)}
            </Typography>
            <Typography color="text.secondary" variant="body2" mt={0.5}>
              {selection.justificacion}
            </Typography>
          </Box>

          <Box className="flex flex-wrap gap-2">
            {selection.regresores_resueltos.length ? (
              selection.regresores_resueltos.map((regressor) => (
                <Chip
                  key={regressor}
                  label={getRegressorDisplayName(regressor)}
                  size="small"
                  variant="outlined"
                  sx={{ fontWeight: 800 }}
                />
              ))
            ) : (
              <Chip label="Sin regresores externos" size="small" variant="outlined" sx={{ fontWeight: 800 }} />
            )}
            <Chip
              label={selection.confiabilidad}
              color={confidenceColor(selection.confiabilidad)}
              size="small"
              sx={{ fontWeight: 800, textTransform: "uppercase" }}
            />
          </Box>

          {selection.advertencia ? (
            <Alert severity="warning">{selection.advertencia}</Alert>
          ) : null}

          <Accordion defaultExpanded={!compact} disableGutters elevation={0} sx={{ bgcolor: "transparent", "&:before": { display: "none" } }}>
            <AccordionSummary expandIcon={<ExpandMoreIcon />} sx={{ px: 0 }}>
              <Typography variant="body2" fontWeight={800} color="text.secondary">
                Ver detalles técnicos
              </Typography>
            </AccordionSummary>
            <AccordionDetails sx={{ px: 0, pb: 0 }}>
              <Box className="grid gap-2 md:grid-cols-3">
                <MiniStat
                  label="MAPE de referencia"
                  value={selection.mape_referencia !== null && selection.mape_referencia !== undefined ? `${formatNumber(selection.mape_referencia)}%` : "Sin dato"}
                  helper={`Origen: ${selection.origen_decision}`}
                />
                <MiniStat
                  label="MAE de referencia"
                  value={selection.mae_referencia !== null && selection.mae_referencia !== undefined ? formatNumber(selection.mae_referencia) : "Sin dato"}
                  helper={selection.folds ? `${selection.folds} folds` : "Sin folds"}
                />
                <MiniStat
                  label="Calibración"
                  value={selection.no_calibrado ? "No calibrado" : "Calibrado"}
                  helper={selection.advertencia || "Selección activa"}
                />
              </Box>

            </AccordionDetails>
          </Accordion>
        </Stack>
      </CardContent>
    </Card>
  );
}

function MiniStat({ label, value, helper }) {
  return (
    <Box className="rounded-xl border border-slate-200 bg-slate-50 p-3">
      <Typography color="text.secondary" variant="body2" fontWeight={800}>
        {label}
      </Typography>
      <Typography component="strong" display="block" mt={0.75} variant="h3" lineHeight={1.15}>
        {value}
      </Typography>
      <Typography color="text.secondary" variant="body2" mt={0.5}>
        {helper}
      </Typography>
    </Box>
  );
}
