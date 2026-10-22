## MODIFIED Requirements

### Requirement: Runtime screen follows main screen prototype
The system SHALL render the runtime inspection screen using the information architecture of the current PySide6 main page and `yolo_action_detection/assets/screenshots/main-window.png`: top status bar, left camera viewport, right KPI and recognition workflow panel, and bottom-right primary controls. Missing HTML prototype files SHALL NOT be required at runtime or build time.

#### Scenario: Runtime screen has prototype regions
- **WHEN** the main window is shown
- **THEN** it SHALL contain a top status bar, a large camera viewport, a right-side KPI area, a recognition progress area, an abnormal notice area, and camera/detection controls

#### Scenario: Configuration remains reachable from top bar
- **WHEN** the operator needs to edit settings from the runtime screen
- **THEN** the configuration action SHALL remain available in the top bar

#### Scenario: Primary controls remain prominent
- **WHEN** the operator views the runtime screen
- **THEN** camera and detection buttons SHALL remain visually prominent and large enough for production use

### Requirement: Recognition progress list replaces large step cards
The system SHALL display recognized categories and quantity progress in a compact recognition progress list. Unrecognized configured categories SHALL stay hidden until they are detected, then appear in detection order.

#### Scenario: Unrecognized items stay hidden
- **WHEN** the runtime screen loads one or more configured categories and none have been recognized yet
- **THEN** the recognition progress area SHALL keep those items hidden

#### Scenario: Empty recognition list
- **WHEN** no valid category is configured
- **THEN** the recognition progress area SHALL show an empty-state message directing the operator to configure categories

#### Scenario: Quantity progress is visible
- **WHEN** the current configured category requires quantity 4 and the current frame recognizes 3
- **THEN** the recognition list SHALL display progress equivalent to `3 / 4`

#### Scenario: Startup loads configured quantities
- **WHEN** the application starts with a configured category quantity greater than 1
- **THEN** the initial step engine SHALL use that configured quantity rather than defaulting to 1

#### Scenario: ROI mode uses exact same-frame quantity
- **WHEN** ROI inspection is enabled and a category requires quantity 4
- **THEN** that category SHALL pass only when exactly 4 matching detections have centers inside the inspection ROI in the same frame

#### Scenario: Completed round holds green state during interval
- **WHEN** a round passes and `round_cooldown_seconds` is greater than 0
- **THEN** the recognition list SHALL keep completed items in the PASS/green state until the configured interval ends

#### Scenario: NG state is visible
- **WHEN** the state machine emits an NG result
- **THEN** the recognition progress area SHALL show the relevant item in an NG/red state and the abnormal notice SHALL be visible

### Requirement: Config screen follows config screen prototype
The system SHALL render the configuration screen using the information architecture of the current PySide6 config page and `yolo_action_detection/assets/screenshots/config-page.png`: left category navigation, right stacked configuration sections, and a fixed footer with cancel and save actions. Missing HTML prototype files SHALL NOT be required.

#### Scenario: Config screen has category navigation
- **WHEN** the operator opens the configuration screen
- **THEN** the screen SHALL show navigation entries for model/steps, action judgement, camera, display/feedback, ROI configuration, and production statistics

#### Scenario: Selecting navigation changes section
- **WHEN** the operator selects a configuration category
- **THEN** the right configuration area SHALL switch to the corresponding form section without losing unsaved field values

#### Scenario: Save actions stay reachable
- **WHEN** the operator scrolls or switches configuration sections
- **THEN** cancel and save actions SHALL remain visible in a fixed footer area

### Requirement: Config sections preserve existing behavior
The redesigned configuration screen SHALL preserve the UI-editable production fields and save behavior while keeping them in the six navigation sections.

#### Scenario: Model and steps section saves categories and counts
- **WHEN** the operator edits model path, thresholds, categories, and quantities then saves
- **THEN** the existing configuration fields SHALL be persisted with the same semantics as before the redesign

#### Scenario: Action judgement section owns round interval
- **WHEN** the operator edits completion interval
- **THEN** the value SHALL be saved to `round_cooldown_seconds` and apply to both normal sequence mode and ROI inspection mode

#### Scenario: ROI section does not duplicate interval
- **WHEN** the operator opens the ROI configuration section
- **THEN** the section SHALL NOT provide a second round interval field

#### Scenario: ROI section uses current locator terminology
- **WHEN** the operator opens ROI configuration
- **THEN** the UI SHALL use inspection ROI, template ROI, and search ROI terminology rather than PCB-specific terminology
