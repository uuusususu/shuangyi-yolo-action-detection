## ADDED Requirements

### Requirement: Runtime screen follows main screen prototype
The system SHALL render the runtime inspection screen using the information architecture from `yolo_action_detection/main-screen.html`: top status bar, left camera viewport, right KPI and recognition workflow panel, and bottom-right primary controls.

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
The system SHALL display configured categories and quantity progress in a compact recognition progress list modeled after `main-screen.html`.

#### Scenario: Configured recognition items are immediately visible
- **WHEN** the runtime screen loads one or more configured categories
- **THEN** the recognition progress area SHALL immediately display every configured category in order, including categories that are still waiting

#### Scenario: Empty recognition list
- **WHEN** no valid category is configured
- **THEN** the recognition progress area SHALL show an empty-state message directing the operator to configure categories

#### Scenario: Quantity progress is visible
- **WHEN** the current configured category requires quantity 4 and the current frame recognizes 3
- **THEN** the recognition list SHALL display progress equivalent to `3 / 4`

#### Scenario: Startup loads configured quantities
- **WHEN** the application starts with a configured category quantity greater than 1
- **THEN** the initial step engine SHALL use that configured quantity rather than defaulting to 1

#### Scenario: Region check uses exact same-frame quantity
- **WHEN** first-category region check is enabled and a child category requires quantity 4
- **THEN** each parent region SHALL pass that child category only when exactly 4 matching children belong to that parent in the same frame

#### Scenario: Completed round holds green state during interval
- **WHEN** a round passes and `round_cooldown_seconds` is greater than 0
- **THEN** the recognition list SHALL keep completed items in the PASS/green state until the configured interval ends

#### Scenario: NG state is visible
- **WHEN** the state machine emits an NG result
- **THEN** the recognition progress area SHALL show the relevant item in an NG/red state and the abnormal notice SHALL be visible

### Requirement: Config screen follows config screen prototype
The system SHALL render the configuration screen using the information architecture from `yolo_action_detection/config-screen.html`: left category navigation, right stacked configuration sections, and a fixed footer with cancel and save actions.

#### Scenario: Config screen has category navigation
- **WHEN** the operator opens the configuration screen
- **THEN** the screen SHALL show navigation entries for model/steps, action judgement, camera, display/feedback, region check, and production statistics

#### Scenario: Selecting navigation changes section
- **WHEN** the operator selects a configuration category
- **THEN** the right configuration area SHALL switch to the corresponding form section without losing unsaved field values

#### Scenario: Save actions stay reachable
- **WHEN** the operator scrolls or switches configuration sections
- **THEN** cancel and save actions SHALL remain visible in a fixed footer area

### Requirement: Config sections preserve existing behavior
The redesigned configuration screen SHALL preserve existing configuration fields and save behavior while reorganizing them into prototype sections.

#### Scenario: Model and steps section saves categories and counts
- **WHEN** the operator edits model path, thresholds, categories, and quantities then saves
- **THEN** the existing configuration fields SHALL be persisted with the same semantics as before the redesign

#### Scenario: Action judgement section owns round interval
- **WHEN** the operator edits completion interval
- **THEN** the value SHALL be saved to `round_cooldown_seconds` and apply to both normal sequence mode and first-category region check mode

#### Scenario: Region section does not duplicate interval
- **WHEN** the operator opens the region check section
- **THEN** the section SHALL NOT provide a second round interval field

#### Scenario: Region section uses generic terminology
- **WHEN** the operator opens region check configuration
- **THEN** the UI SHALL use parent-region and child-control terminology rather than PCB-specific terminology

### Requirement: UI quality constraints
The redesigned UI SHALL meet production usability constraints for dark industrial desktop operation.

#### Scenario: Inputs have visible labels
- **WHEN** a form input is shown
- **THEN** it SHALL have a visible label and, where needed, helper text

#### Scenario: Interactive targets are large enough
- **WHEN** a button, toggle, or navigation item is shown
- **THEN** it SHALL have at least production-friendly hit size comparable to 40px height, with primary controls larger

#### Scenario: Save feedback is visible
- **WHEN** configuration is saved successfully
- **THEN** the UI SHALL show a short visible saved confirmation

#### Scenario: Dangerous reset has confirmation
- **WHEN** the operator triggers statistics reset/archive
- **THEN** the UI SHALL show a confirmation dialog before changing statistics
