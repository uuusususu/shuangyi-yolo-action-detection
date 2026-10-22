## 1. Tests first

- [x] 1.1 Change template-save tests so a template can be saved without any search ROI, and a small or disjoint search draft does not block template save.
- [x] 1.2 Add tests that a fresh config page shows no ROI boxes, and create-or-replace does not persist a rectangle until save-current-ROI.

## 2. Decouple save and verify

- [x] 2.1 Save template ROI by cropping, building the model, and verifying in an expanded template ROI or the full static frame, never in the search ROI.
- [x] 2.2 Stop rejecting search-ROI save because of travel headroom; keep that check on the start-detection path only.

## 3. Empty until created

- [x] 3.1 Keep create-or-replace as an editor draft only, and load only previously saved rectangles when opening the page.

## 4. Verify

- [x] 4.1 Run the ROI config tests plus `openspec validate decouple-template-create-from-search-roi --strict`.
