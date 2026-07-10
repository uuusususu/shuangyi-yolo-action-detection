## ADDED Requirements

### Requirement: Step quantity configuration
The system SHALL allow each configured action step to define a required quantity for its category, with a default required quantity of 1 for existing or missing configuration values.

#### Scenario: Existing configuration without quantities
- **WHEN** the application loads a configuration that contains `category_names` but no `category_counts`
- **THEN** each step SHALL use required quantity 1 by default

#### Scenario: Operator configures category quantity
- **WHEN** the operator sets step 1 category to `脚垫` and quantity to 4
- **THEN** the saved runtime configuration SHALL preserve step 1 as category `脚垫` with required quantity 4

#### Scenario: Empty category ignores quantity
- **WHEN** a step category is empty
- **THEN** the step SHALL remain unconfigured regardless of the quantity input value

#### Scenario: Invalid quantity is normalized or rejected
- **WHEN** a configured step has quantity less than 1
- **THEN** the system SHALL prevent that configured step from running with a quantity below 1

### Requirement: Single-frame quantity matching
The system SHALL evaluate the current action step by counting detections of that step's category in the current frame only, and SHALL pass the step only when the current frame count equals the required quantity.

#### Scenario: Current frame matches required quantity
- **WHEN** the current step is category `脚垫` with required quantity 4 and the current frame contains exactly 4 `脚垫` detections
- **THEN** the current step SHALL satisfy its quantity condition for that frame

#### Scenario: Current frame below required quantity
- **WHEN** the current step is category `脚垫` with required quantity 4 and the current frame contains 3 `脚垫` detections
- **THEN** the current step SHALL NOT pass from that frame

#### Scenario: Current frame above required quantity
- **WHEN** the current step is category `脚垫` with required quantity 4 and the current frame contains 5 `脚垫` detections
- **THEN** the current step SHALL NOT pass from that frame

#### Scenario: Counts do not accumulate across frames
- **WHEN** the current step requires 4 `脚垫` detections, one frame contains 2 `脚垫` detections, and the next frame contains 2 `脚垫` detections
- **THEN** the current step SHALL NOT pass by adding the two frames together

### Requirement: Quantity-aware step progression
The system SHALL advance to the next configured action step only after the current step satisfies its quantity condition and existing pass stability requirements.

#### Scenario: One matching frame passes when stability is one
- **WHEN** the current step requires 4 `脚垫` detections and pass stability is 1 frame
- **THEN** one frame with exactly 4 `脚垫` detections SHALL pass the current step and move to the next configured step

#### Scenario: Stability counts matching frames, not objects
- **WHEN** the current step requires 4 `脚垫` detections and pass stability is 2 frames
- **THEN** the step SHALL pass only after two consecutive frames each contain exactly 4 `脚垫` detections

#### Scenario: Current step pass does not cascade in same frame
- **WHEN** one frame contains the exact required quantity for the current step and the exact required quantity for the next step
- **THEN** the system SHALL pass only the current step during that update and SHALL evaluate the next step on a later update

### Requirement: Quantity progress display
The system SHALL display the current frame count and required count for configured action steps so the operator can see quantity progress.

#### Scenario: Current step displays progress
- **WHEN** the current step is `脚垫 × 4` and the current frame contains 3 `脚垫` detections
- **THEN** the main step display SHALL show progress equivalent to `3/4`

#### Scenario: Passed step preserves matched progress
- **WHEN** a step passes after detecting the required quantity
- **THEN** the step display SHALL retain a passed state and show the matched quantity context for that step
