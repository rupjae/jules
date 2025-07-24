import { useEffect, useState } from "react";
import Switch from "@mui/material/Switch";

// Local-storage key so the preference survives page reloads.
const KEY = "showInfoPacket";

/**
 * Toggle that lets power-users decide whether the retrieval info-packet
 * (background notes) should be rendered under assistant messages.
 *
 * The component is *self-contained* – it synchronises its state with
 * `localStorage` and therefore has no external props.
 */
export default function InfoPacketToggle() {
  const [checked, setChecked] = useState(false);

  // Hydrate from localStorage on mount (client-side only).
  useEffect(() => {
    if (typeof window === "undefined") return;
    setChecked(localStorage.getItem(KEY) === "true");
  }, []);

  return (
    <label style={{ cursor: "pointer" }}>
      Show Retrieval Info-Packet{" "}
      <Switch
        checked={checked}
        onChange={(e) => {
          const next = e.target.checked;
          setChecked(next);
          if (typeof window !== "undefined") {
            localStorage.setItem(KEY, String(next));
          }
        }}
      />
    </label>
  );
}

