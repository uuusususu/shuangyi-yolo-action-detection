## ADDED Requirements

### Requirement: Production controls stay wired to live runtime objects
The system SHALL keep the redesigned runtime and configuration controls connected to the current camera worker, processors, step engine, ROI engine, sound feedback, fail evidence, and production statistics. Deleted PCB, first-category region check, calibration, and hole-judgement modules SHALL NOT be reattached to these controls.

#### Scenario: Runtime primary actions still start camera and detection
- **WHEN** the operator clicks the runtime camera and detection buttons
- **THEN** those buttons SHALL call the existing `CameraWorker` and `AppState` start/stop path and SHALL NOT instantiate deleted PCB or first-category engines

#### Scenario: Config save applies live production fields only
- **WHEN** the operator clicks validate-and-save
- **THEN** the page SHALL persist the UI-editable production fields and the main window SHALL reload model, camera SN, step engine, ROI engine, and feedback settings from that saved config

#### Scenario: Independent ROI save is unchanged
- **WHEN** the operator saves the currently selected ROI
- **THEN** only that ROI kind SHALL be written and the full-page save SHALL NOT overwrite the other two stored ROI rectangles with drafts

#### Scenario: Back without save does not write the form
- **WHEN** the operator edits unsaved fields and clicks return-to-detection
- **THEN** `ConfigManager` SHALL keep the last saved values and the runtime engines SHALL refresh from those saved values only

### Requirement: Control inventory separates live, themed, and retired widgets
The production UI SHALL use only the live redesigned widgets. Retired widgets from the pre-redesign shell SHALL NOT be constructed on the runtime or configuration pages.

#### Scenario: Live widgets are the only production widgets
- **WHEN** the main window and configuration page are shown
- **THEN** they SHALL use `KpiRow`, `RecognitionListItem`, `ConfigNavButton`, `FormGroup`, `FormField`, `SwitchRow`, `FooterActionBar`, and `RoiEditorWidget` for their respective roles

#### Scenario: Retired widgets stay off the production path
- **WHEN** the main window or configuration page is constructed
- **THEN** they SHALL NOT instantiate `StepCard`, `PanelSection`, `MetricCard`, or `DefectChip`

#### Scenario: Theme owns standard control chrome
- **WHEN** a standard button, text field, combo box, spin box, or checkbox is shown
- **THEN** PyQtDarkTheme SHALL own its border and interaction states while project tokens SHALL own panel backgrounds and PASS/NG colors

### Requirement: Config fields have an explicit UI boundary
The configuration page SHALL expose only the classified UI-editable production fields. File-only compatibility fields MAY remain in `ConfigManager` for load/save, and stripped historical keys SHALL stay absent from both UI and portable config.

#### Scenario: UI-editable fields round-trip
- **WHEN** the operator changes model path, categories, counts, action frames, cooldown, camera SN, overlay/sound/evidence switches, ROI mode switches, template thresholds, and today target capacity then saves
- **THEN** those values SHALL reload into the same controls after reopening the configuration page

#### Scenario: Legacy judgement fields stay file-only
- **WHEN** the operator opens any configuration section
- **THEN** the page SHALL NOT show calibration points, holes, torque-gun, fusion, or joint-tracking fields

#### Scenario: Deleted inspection keys stay stripped
- **WHEN** portable config is prepared
- **THEN** it SHALL continue omitting `pcb_inspection_*` and `first_category_region_check_*`

### Requirement: Recognition items appear only after detection
The recognition list SHALL stay empty of unrecognized categories. A configured item SHALL appear only after it is recognized, then later items SHALL be appended in detection order. Engine updates SHALL change state, quantity, and visibility together.

#### Scenario: Waiting sequence items stay hidden
- **WHEN** normal sequence mode starts with configured categories and none have been recognized yet
- **THEN** every configured item SHALL remain hidden

#### Scenario: Items appear in detection order
- **WHEN** the first configured category is recognized and later categories are still unrecognized
- **THEN** only the recognized item SHALL be visible in the recognition list

#### Scenario: ROI items stay hidden before locator trigger
- **WHEN** ROI mode is waiting for a locator or inspection trigger
- **THEN** every configured inspection category SHALL remain hidden

#### Scenario: Zero quantity stays hidden
- **WHEN** a configured item still has quantity `0 / N` and is not PASS or NG
- **THEN** the item SHALL stay hidden

### Requirement: KPI capacity shows inspected count against the target
The runtime KPI labeled as today's capacity SHALL show the current batch inspected count together with the configured `today_target_capacity`.

#### Scenario: Capacity displays inspected over target
- **WHEN** the current batch has inspected 12 pieces and `today_target_capacity` is 300
- **THEN** the capacity KPI SHALL display a value equivalent to `12 / 300`

#### Scenario: Target change appears after save
- **WHEN** the operator saves a new today target capacity and returns to the runtime screen
- **THEN** the capacity KPI SHALL use the newly saved target with the unchanged batch inspected count
