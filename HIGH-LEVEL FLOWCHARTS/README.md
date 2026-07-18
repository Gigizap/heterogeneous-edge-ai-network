# High-Level Flowcharts - Notation Guide

This folder holds the seven flowcharts that describe the full runtime behaviour of the
architecture, plus the LaTeX appendix that embeds them ([appendix_flowcharts.tex](appendix_flowcharts.tex)).

The diagrams (in `figures/`, read in this order):

1. `1_startup.png` - device startup
2. `2_discovery.png` - peer discovery loop
3. `3_peer_joined.png` - `peer_joined(peer)` handler
4. `4_peer_lost.png` - `peer_lost(peer)` handler
5. `5_leader_electionpng.png` - leader election
6. `6_polling_query.png` - leader query-polling loop
7. `7_query_reply.png` - `query_reply(query)` procedure

This file records which symbols follow standard flowchart convention (ISO 5807 / ANSI)
and which do not, so a reader (or reviewer) can decode the diagrams and so you can decide
what, if anything, to change before publication.

## Shape legend

| Symbol in the diagrams | Meaning here | Standard? |
|---|---|---|
| Ellipse / oval | Terminator (entry point or "Done") | Yes - standard terminator |
| Rounded rectangle | Process / action step | Yes - process (see note 3) |
| Diamond | Decision / branch | Yes - standard decision |
| Parallelogram | Input/Output: data sent to or received from another node | Yes - standard I/O ("Data") symbol |
| Coloured box with "(see X diagram)" text | Call into another flowchart (a subroutine) | No - see item 1 below |
| Green box, red dashed border ("tool list", "Merge into tool list", ...) | The shared tool-list data store, and reads/writes on it | No - see item 2 below |
| Dashed "reads" arrow (Fig. 7) | Data access (read) rather than control flow | Borderline - see item 2 below |

## What already follows convention

- **Ellipse = terminator.** Start ovals (`Discovery loop`, `peer_joined( peer )`,
  `query_reply( query )`, ...) and every `Done` are correct terminators.
- **Rectangle = process.** Action steps such as `Load agent presets`, `Connect agent A2A`,
  `Identify lost peer`, `Reset peer timer` are correct process boxes.
- **Diamond = decision.** `Event type?`, `Known peer?`, `Was it the leader?`,
  `Backed-up query found?`, `Query arrived?` are correct decisions.
- **Parallelogram = Input/Output.** This is the standard "Data" / I/O symbol, and using it
  for data moving in or out of a node is a correct, conventional use. It is applied
  consistently to the network messages: `Send heartbeat`, `heartbeat received`,
  `Leader receives tools`, `Agents send scores`, `approach 1 agents send tools`,
  `Tool results to leader`, `Back up query to backup node`, `Remove query from backup node`.
  **No change needed here.**

## What is NOT conventional, and what to change

### 1. Colour + "(see X diagram)" used as a subroutine call
A coloured box whose text reads "(see ... diagram)" means "control jumps to that other
flowchart", and its colour matches the start terminator of the target diagram
(purple -> discovery, blue -> peer_joined, pink -> peer_lost, red -> leader_election,
teal -> query_reply, yellow -> polling). This colour-as-link is a custom convention:
colour carries meaning, and readers printing in greyscale lose it.

- **Conventional symbol:** the **predefined process** (a rectangle with a double vertical
  bar on each side) denotes a call to a separately defined process.
- **What to change:** draw those call boxes as predefined-process rectangles, and keep the
  "(see ... diagram)" text as the reference. Colour can stay as a visual aid but should not
  be the only thing that signals a subroutine call.

### 2. Green dashed box = the tool-list data store
The green, red-dashed box is the leader's merged tool list: a stored data structure that
steps write (`Merge into tool list`, `Remove tool from tool list`) and read (the dashed
`reads` arrow in Fig. 7, into `LLM chooses the tool`). A rounded rectangle is not the
symbol for stored data.

- **Conventional symbol:** a **cylinder** (database / data store) or the "stored data"
  symbol for the tool list itself; the write/read steps stay as ordinary process boxes that
  point to or from that store.
- **What to change:** replace the standalone `Tool list` box in Fig. 7 with a cylinder, and
  either (a) turn `Merge into tool list` / `Remove tool from tool list` into normal process
  boxes with an arrow to the cylinder, or (b) keep them but drop the dashed border so they
  read as ordinary processes. The dashed `reads` arrow is acceptable to mark a data access,
  but note it in the legend if you keep it.

### 3. Process boxes are rounded rectangles (minor, no change needed)
Every process box in every chart is a rounded rectangle, applied consistently. Strict
ISO 5807 / ANSI draws a process as a plain (sharp-cornered) rectangle and reserves the
rounded shape for terminators, but since the terminators here are ellipses there is no
ambiguity, so this is safe to leave as is. The only variation is the corner radius: the
smaller boxes in Fig. 7 are drawn with a tighter radius than the larger boxes, which is a
side effect of box size, not a different symbol. Nothing to change.

### 4. Dashed "loop back" box (Fig. 2)
The dashed `loop back to await event` box in Fig. 2 is an annotation, not a step.
Replace it with a labelled arrow back to `Await event`, or use an on-page **connector**
(a small circle) if the return arrow would cross the diagram.

## Recommendation

If strict ISO 5807 / ANSI compliance is required for the target venue, apply changes 1 and 2;
they are the only two places where a reader could misread the diagram without the colour
key. Items 3 and 4 are cosmetic and optional.

If strict compliance is not required, the diagrams are already readable as long as the legend
travels with them. That legend is included in the appendix intro paragraph in
[appendix_flowcharts.tex](appendix_flowcharts.tex), so no diagram edit is strictly necessary
to publish.
