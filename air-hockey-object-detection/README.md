# Air Hockey Object Detection

Track a puck and two strikers through footage of an air hockey game using
classical computer vision only — no neural networks anywhere. The project works
through a progression, from basic pixel operations to detection that survives
fast motion, and renders the whole thing as a single annotated video.

The output is built as a **narrated demonstration**: the processing applied to
each frame is chosen by that frame's timestamp, so as the video plays it moves
through one technique after another, with on-screen subtitles naming the
technique, explaining what it does, and printing the parameters in use. Watching
`processed-output.mp4` end to end is the fastest way to understand the project.

## What it covers

**Basic image processing.** Grayscale conversion, Gaussian and bilateral
blurring — the latter chosen where edges need to survive the smoothing — and
Sobel gradients computed separately in the horizontal and vertical directions
before being combined into a single edge response.

**Colour-based object isolation.** The puck and both strikers are separated by
converting to **HSV** and thresholding on hue, which holds up under the changing
brightness across the table far better than working in RGB. The resulting masks
are cleaned with morphological opening to drop speckle noise, then closing to
fill gaps, leaving blobs solid enough to fit a shape to.

**Circle detection.** A **Hough circle transform** finds the round pieces. Two
configurations are kept: a general one for the strikers, and a tighter one for
the puck with a narrower radius band, since a single parameter set could not
handle both reliably.

**Template matching.** A puck template is extracted from the footage itself and
matched with normalised cross-correlation. This is the more robust path once the
puck is moving quickly, where motion blur weakens the circular edge that Hough
depends on — the two methods fail in different conditions, which is the point of
implementing both.

**Effects driven by the detections.** Because the puck's position is known per
frame, the pipeline can act on it: recolouring the puck by shifting its hue
within the mask, drawing a fading motion trail from its recent positions, and
firing a radiating flash on collisions, with a cooldown so a single impact does
not retrigger across consecutive frames.

## Files

```
air-hockey-object-detection.ipynb   the pipeline, section by section
video.mp4                           source footage, 848x478, 30fps, ~63s
processed-output.mp4                the rendered result with subtitles
```

## Running it

Open the notebook and run it top to bottom. It reads `video.mp4` from the working
directory — that exact filename is set in the `INPUT_FILE_NAME` constant in the
configuration cell, which is why the source footage is committed under that name
rather than a more descriptive one. Rendering the full video takes a few minutes
and writes a new file; the committed `processed-output.mp4` is a previous render
kept so the result can be watched without running anything.

To inspect a single frame instead of rendering the whole video, set
`SHOW_FRAME_AT` to a timestamp in milliseconds.
