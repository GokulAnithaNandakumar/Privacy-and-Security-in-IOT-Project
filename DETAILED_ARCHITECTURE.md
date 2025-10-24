===============================================================================
                    SYSTEM ARCHITECTURE - DETAILED DIAGRAMS
                IoT PRIVACY & SECURITY SYSTEM WITH BLOCKCHAIN INTEGRATION
===============================================================================

### DETAILED SYSTEM ARCHITECTURE DIAGRAMS

#### 3.1 High-Level System Block Diagram

```mermaid
graph TB
    subgraph "IoT SENSOR LAYER"
        A1[Living Room Sensors]
        A2[Bedroom Sensors]
        A3[Kitchen Sensors]
        A4[Office Sensors]
        A5[File Upload Interface]
    end

    subgraph "DATA COLLECTION LAYER"
        B1[Sensor Data Collector]
        B2[Data Aggregator]
        B3[Real-time Buffer]
    end

    subgraph "PRIVACY PRESERVATION LAYER"
        C1[Differential Privacy<br/>ε-mechanism]
        C2[K-Anonymity<br/>Processor]
        C3[AES-256-CBC<br/>Encryption]
        C4[Data Validator]
    end

    subgraph "BLOCKCHAIN LAYER"
        D1[Smart Contract<br/>Interface]
        D2[Transaction<br/>Manager]
        D3[Ethereum<br/>Network]
        D4[Block Storage]
    end

    subgraph "MACHINE LEARNING LAYER"
        E1[Feature<br/>Extractor]
        E2[Training<br/>Controller]
        E3[Isolation Forest<br/>Model]
        E4[Anomaly<br/>Detector]
    end

    subgraph "PRESENTATION LAYER"
        F1[Dashboard<br/>Controller]
        F2[Blockchain<br/>Visualizer]
        F3[File<br/>Manager]
        F4[Alert<br/>System]
    end

    A1 --> B1
    A2 --> B1
    A3 --> B1
    A4 --> B1
    A5 --> B2

    B1 --> B2
    B2 --> B3
    B3 --> C1

    C1 --> C2
    C2 --> C3
    C3 --> C4
    C4 --> D1

    D1 --> D2
    D2 --> D3
    D3 --> D4
    D4 --> E1

    E1 --> E2
    E2 --> E3
    E3 --> E4
    E4 --> F1

    F1 --> F2
    F2 --> F3
    F3 --> F4

    style A1 fill:#e1f5fe
    style A2 fill:#e1f5fe
    style A3 fill:#e1f5fe
    style A4 fill:#e1f5fe
    style C1 fill:#fff3e0
    style C2 fill:#fff3e0
    style C3 fill:#fff3e0
    style D3 fill:#e8f5e8
    style E3 fill:#fce4ec
```

#### 3.2 Data Flow Architecture

```mermaid
graph LR
    subgraph "INPUT SOURCES"
        S1[Temperature<br/>Sensors]
        S2[Humidity<br/>Sensors]
        S3[Motion<br/>Sensors]
        S4[Light<br/>Sensors]
        S5[Air Quality<br/>Sensors]
        S6[File<br/>Uploads]
    end

    subgraph "PRIVACY PIPELINE"
        P1[Raw Data<br/>Collection]
        P2[Differential<br/>Privacy<br/>ε=1.0]
        P3[K-Anonymity<br/>k=3]
        P4[AES-256<br/>Encryption]
        P5[Data<br/>Validation]
    end

    subgraph "BLOCKCHAIN PROCESSING"
        B1[Smart Contract<br/>Processing]
        B2[Gas<br/>Optimization]
        B3[Transaction<br/>Creation]
        B4[Block<br/>Mining]
        B5[Immutable<br/>Storage]
    end

    subgraph "ML ANALYSIS"
        M1[Data Buffer<br/>≥10 points]
        M2[Auto Training<br/>Trigger]
        M3[Feature<br/>Engineering]
        M4[Isolation Forest<br/>Training]
        M5[Real-time<br/>Detection]
    end

    subgraph "USER INTERFACE"
        U1[Real-time<br/>Dashboard]
        U2[Blockchain<br/>Visualization]
        U3[Transaction<br/>Explorer]
        U4[File<br/>Management]
        U5[Security<br/>Alerts]
    end

    S1 --> P1
    S2 --> P1
    S3 --> P1
    S4 --> P1
    S5 --> P1
    S6 --> P1

    P1 --> P2
    P2 --> P3
    P3 --> P4
    P4 --> P5
    P5 --> B1

    B1 --> B2
    B2 --> B3
    B3 --> B4
    B4 --> B5
    B5 --> M1

    M1 --> M2
    M2 --> M3
    M3 --> M4
    M4 --> M5
    M5 --> U1

    U1 --> U2
    U2 --> U3
    U3 --> U4
    U4 --> U5
```

#### 3.3 Component Interaction Diagram

```mermaid
graph TB
    subgraph "PHYSICAL LAYER"
        direction TB
        IOT[IoT Device Network<br/>4 Rooms × 5 Sensor Types]
        NET[Network Interface<br/>WiFi/Ethernet]
    end

    subgraph "APPLICATION LAYER"
        direction TB
        subgraph "Core Processing"
            DC[Data Collector<br/>Real-time Streaming]
            PP[Privacy Processor<br/>Multi-layer Protection]
            BM[Blockchain Manager<br/>Smart Contract Interface]
        end

        subgraph "Analytics Engine"
            ML[ML Engine<br/>Isolation Forest]
            AD[Anomaly Detector<br/>Real-time Analysis]
            AS[Alert System<br/>Security Monitoring]
        end

        subgraph "Storage Systems"
            BC[Blockchain Storage<br/>Immutable Records]
            FS[File Storage<br/>Encrypted Files]
            BF[Buffer Storage<br/>Temporary Data]
        end
    end

    subgraph "PRESENTATION LAYER"
        direction TB
        WEB[Web Dashboard<br/>Streamlit Interface]
        VIZ[Visualization Engine<br/>Plotly Graphics]
        API[REST API<br/>External Access]
    end

    subgraph "EXTERNAL SERVICES"
        direction TB
        ETH[Ethereum Network<br/>Blockchain Infrastructure]
        IPFS[IPFS<br/>Distributed Storage]
        EXT[External APIs<br/>Third-party Services]
    end

    IOT --> NET
    NET --> DC
    DC --> PP
    PP --> BM
    BM --> BC
    BC --> ML
    ML --> AD
    AD --> AS

    DC --> BF
    BF --> ML

    PP --> FS
    FS --> VIZ

    AS --> WEB
    BC --> VIZ
    VIZ --> WEB
    WEB --> API

    BM --> ETH
    FS --> IPFS
    API --> EXT

    style IOT fill:#e3f2fd
    style PP fill:#fff8e1
    style BC fill:#e8f5e8
    style ML fill:#fce4ec
    style WEB fill:#f3e5f5
```

#### 3.4 Security Architecture

```mermaid
graph TB
    subgraph "SECURITY LAYERS"
        direction TB

        subgraph "INPUT SECURITY"
            IS1[Input Validation<br/>Data Sanitization]
            IS2[Rate Limiting<br/>DDoS Protection]
            IS3[Authentication<br/>Access Control]
        end

        subgraph "PRIVACY SECURITY"
            PS1[Differential Privacy<br/>ε-mechanism Protection]
            PS2[K-Anonymity<br/>Identity Protection]
            PS3[Data Masking<br/>Sensitive Field Protection]
        end

        subgraph "ENCRYPTION SECURITY"
            ES1[AES-256-CBC<br/>Data Encryption]
            ES2[Key Management<br/>Secure Key Storage]
            ES3[Hash Functions<br/>Data Integrity]
        end

        subgraph "BLOCKCHAIN SECURITY"
            BS1[Smart Contract<br/>Audit & Verification]
            BS2[Transaction<br/>Digital Signatures]
            BS3[Consensus<br/>Network Security]
        end

        subgraph "ML SECURITY"
            MS1[Model Protection<br/>Anti-tampering]
            MS2[Adversarial<br/>Attack Detection]
            MS3[Confidence<br/>Threshold Validation]
        end

        subgraph "APPLICATION SECURITY"
            AS1[Web Security<br/>HTTPS/TLS]
            AS2[Session Management<br/>Secure Sessions]
            AS3[Audit Logging<br/>Security Monitoring]
        end
    end

    IS1 --> PS1
    IS2 --> PS2
    IS3 --> PS3

    PS1 --> ES1
    PS2 --> ES2
    PS3 --> ES3

    ES1 --> BS1
    ES2 --> BS2
    ES3 --> BS3

    BS1 --> MS1
    BS2 --> MS2
    BS3 --> MS3

    MS1 --> AS1
    MS2 --> AS2
    MS3 --> AS3

    style IS1 fill:#ffebee
    style PS1 fill:#fff3e0
    style ES1 fill:#e8f5e8
    style BS1 fill:#e3f2fd
    style MS1 fill:#fce4ec
    style AS1 fill:#f3e5f5
```

#### 3.5 Complete System Integration Architecture

```mermaid
graph TB
    subgraph "IoT SENSOR ECOSYSTEM"
        direction TB
        subgraph "Living Room"
            LR1[Temperature]
            LR2[Humidity]
            LR3[Motion]
            LR4[Light]
            LR5[Air Quality]
        end
        subgraph "Bedroom"
            BR1[Temperature]
            BR2[Humidity]
            BR3[Motion]
            BR4[Light]
            BR5[Air Quality]
        end
        subgraph "Kitchen"
            KR1[Temperature]
            KR2[Humidity]
            KR3[Motion]
            KR4[Light]
            KR5[Air Quality]
        end
        subgraph "Office"
            OR1[Temperature]
            OR2[Humidity]
            OR3[Motion]
            OR4[Light]
            OR5[Air Quality]
        end
    end

    subgraph "DATA PROCESSING PIPELINE"
        direction TB
        DC[Data Collector<br/>Real-time Streaming]
        DA[Data Aggregator<br/>Multi-sensor Fusion]
        DV[Data Validator<br/>Quality Assurance]
    end

    subgraph "PRIVACY PRESERVATION ENGINE"
        direction TB
        DP[Differential Privacy<br/>ε-mechanism]
        KA[K-Anonymity<br/>k=3 Grouping]
        AES[AES-256-CBC<br/>Encryption]
        DM[Data Masking<br/>Sensitive Fields]
    end

    subgraph "BLOCKCHAIN INFRASTRUCTURE"
        direction TB
        SC[Smart Contract<br/>Data Storage Logic]
        TM[Transaction Manager<br/>Gas Optimization]
        BV[Block Validator<br/>Consensus Verification]
        IS[Immutable Storage<br/>Permanent Records]
    end

    subgraph "MACHINE LEARNING CORE"
        direction TB
        FE[Feature Extractor<br/>Data Preprocessing]
        TC[Training Controller<br/>Auto-trigger ≥10]
        IF[Isolation Forest<br/>Anomaly Model]
        RT[Real-time Detector<br/>Live Analysis]
    end

    subgraph "USER INTERFACE LAYER"
        direction TB
        MD[Main Dashboard<br/>System Overview]
        BV2[Blockchain Viewer<br/>Transaction Analysis]
        FM2[File Manager<br/>Storage Interface]
        TE[Transaction Explorer<br/>Detailed Search]
        SS[System Settings<br/>Configuration]
    end

    %% IoT to Data Processing
    LR1 --> DC
    LR2 --> DC
    LR3 --> DC
    LR4 --> DC
    LR5 --> DC
    BR1 --> DC
    BR2 --> DC
    BR3 --> DC
    BR4 --> DC
    BR5 --> DC
    KR1 --> DC
    KR2 --> DC
    KR3 --> DC
    KR4 --> DC
    KR5 --> DC
    OR1 --> DC
    OR2 --> DC
    OR3 --> DC
    OR4 --> DC
    OR5 --> DC

    %% Data Processing Pipeline
    DC --> DA
    DA --> DV

    %% Privacy Pipeline
    DV --> DP
    DP --> KA
    KA --> AES
    AES --> DM

    %% Blockchain Processing
    DM --> SC
    SC --> TM
    TM --> BV
    BV --> IS

    %% ML Pipeline
    IS --> FE
    FE --> TC
    TC --> IF
    IF --> RT

    %% User Interface
    IS --> MD
    MD --> BV2
    BV2 --> FM2
    FM2 --> TE
    TE --> SS

    style LR1 fill:#e1f5fe
    style DP fill:#fff3e0
    style SC fill:#e8f5e8
    style IF fill:#fce4ec
    style MD fill:#f3e5f5
```

===============================================================================
END OF ARCHITECTURE DIAGRAMS
===============================================================================