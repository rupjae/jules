import Accordion from '@mui/material/Accordion';
import AccordionSummary from '@mui/material/AccordionSummary';
import AccordionDetails from '@mui/material/AccordionDetails';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import Typography from '@mui/material/Typography';

export interface RetrievalInfoProps {
  need_search: boolean;
  info_packet: string | null;
}

export default function RetrievalInfo({ need_search, info_packet }: RetrievalInfoProps) {
  return (
    <Accordion sx={{ my: 1 }} defaultExpanded={false} data-testid="retrieval-info">
      <AccordionSummary expandIcon={<ExpandMoreIcon />}>
        <Typography variant="caption" sx={{ fontStyle: 'italic' }}>
          Retrieval considered helpful: {need_search ? 'Yes' : 'No'}
        </Typography>
      </AccordionSummary>
      {need_search && info_packet && (
        <AccordionDetails>
          <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>{info_packet}</pre>
        </AccordionDetails>
      )}
    </Accordion>
  );
}

