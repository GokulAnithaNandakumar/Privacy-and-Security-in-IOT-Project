# IoT Privacy & Security System - Complete Architecture

## 1. System Architecture Flowchart

```mermaid
graph TB
    A[IoT Sensors] --> B[Data Collection]
    B --> C{Privacy Pipeline}
    C --> D[Differential Privacy]
    C --> E[K-Anonymity]
    C --> F[AES Encryption]

    D --> G[Blockchain Storage]
    E --> G
    F --> G

    G --> H[Smart Contract]
    H --> I[Transaction Hash]
    I --> J[Immutable Storage]

    J --> K[ML Pipeline]
    K --> L{Data Sufficient?}
    L -->|≥10 points| M[Auto-Train Model]
    L -->|<10 points| N[Collect More Data]
    N --> B

    M --> O[Isolation Forest]
    O --> P[Anomaly Detection]
    P --> Q{Intrusion Detected?}
    Q -->|Yes| R[Security Alert]
    Q -->|No| S[Normal Operation]

    R --> T[Dashboard Alert]
    S --> U[Continue Monitoring]

    J --> V[File Storage]
    V --> W[Encryption Tracking]
    W --> X[Metadata Management]

    T --> Y[Streamlit Dashboard]
    U --> Y
    X --> Y

    Y --> Z[User Interface]
    Z --> AA[Blockchain Visualization]
    Z --> BB[Transaction Explorer]
    Z --> CC[System Settings]
    Z --> DD[File Management]

    AA --> EE[Real-time Analytics]
    BB --> FF[Search & Filter]
    CC --> GG[Privacy Configuration]
    DD --> HH[Upload & Download]
```

## 2. Layered Architecture Block Diagram

```mermaid
graph TB
    subgraph "PRESENTATION LAYER"
        P1[Streamlit Dashboard]
        P2[Interactive UI]
        P3[Real-time Visualization]
        P4[Alert Management]
    end

    subgraph "APPLICATION LAYER"
        A1[IoT Data Manager]
        A2[Privacy Controller]
        A3[Blockchain Interface]
        A4[ML Engine]
        A5[File Manager]
    end

    subgraph "BUSINESS LOGIC LAYER"
        B1[Privacy Algorithms]
        B2[Smart Contracts]
        B3[ML Models]
        B4[Security Policies]
        B5[Data Validation]
    end

    subgraph "DATA ACCESS LAYER"
        D1[Blockchain Storage]
        D2[File Storage]
        D3[Configuration Store]
        D4[Temporary Buffers]
    end

    subgraph "INFRASTRUCTURE LAYER"
        I1[Ethereum Network]
        I2[IPFS Network]
        I3[Local Storage]
        I4[Network Interface]
    end

    P1 --> A1
    P2 --> A2
    P3 --> A3
    P4 --> A4

    A1 --> B1
    A2 --> B2
    A3 --> B3
    A4 --> B4
    A5 --> B5

    B1 --> D1
    B2 --> D2
    B3 --> D3
    B4 --> D4
    B5 --> D1

    D1 --> I1
    D2 --> I2
    D3 --> I3
    D4 --> I4

    style P1 fill:#e3f2fd
    style A1 fill:#fff8e1
    style B1 fill:#e8f5e8
    style D1 fill:#fce4ec
    style I1 fill:#f3e5f5
```## 3. Complete System Integration Architecture

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

    subgraph "SECURITY MONITORING"
        direction TB
        AD[Anomaly Detector<br/>ML-based Analysis]
        TA[Threat Analyzer<br/>Risk Assessment]
        AS[Alert System<br/>Notification Engine]
        IR[Incident Reporter<br/>Log Management]
    end

    subgraph "FILE MANAGEMENT SYSTEM"
        direction TB
        FU[File Uploader<br/>Multi-format Support]
        FE2[File Encryptor<br/>Secure Storage]
        FM[Metadata Manager<br/>File Tracking]
        FR[File Retriever<br/>Download Service]
    end

    subgraph "USER INTERFACE LAYER"
        direction TB
        MD[Main Dashboard<br/>System Overview]
        BV2[Blockchain Viewer<br/>Transaction Analysis]
        FM2[File Manager<br/>Storage Interface]
        TE[Transaction Explorer<br/>Detailed Search]
        SS[System Settings<br/>Configuration]
    end

    subgraph "EXTERNAL INTEGRATIONS"
        direction TB
        EN[Ethereum Network<br/>Public Blockchain]
        IPFS[IPFS Network<br/>Distributed Storage]
        API[External APIs<br/>Third-party Services]
        WEB[Web Services<br/>Remote Access]
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

    %% Security Monitoring
    RT --> AD
    AD --> TA
    TA --> AS
    AS --> IR

    %% File Management
    FU --> FE2
    FE2 --> FM
    FM --> FR
    FR --> IS

    %% User Interface
    IS --> MD
    MD --> BV2
    BV2 --> FM2
    FM2 --> TE
    TE --> SS

    %% External Integrations
    BV --> EN
    FM --> IPFS
    SS --> API
    MD --> WEB

    %% Alert Feedback
    AS --> MD
    IR --> BV2

    style LR1 fill:#e1f5fe
    style DP fill:#fff3e0
    style SC fill:#e8f5e8
    style IF fill:#fce4ec
    style MD fill:#f3e5f5
    style EN fill:#ede7f6
```

## 4. Deployment Architecture

```mermaid
graph TB
    subgraph "DEVELOPMENT ENVIRONMENT"
        DEV1[Local Development<br/>Python 3.11+]
        DEV2[Conda Environment<br/>Dependency Management]
        DEV3[VS Code IDE<br/>Development Tools]
        DEV4[Git Version Control<br/>Source Management]
    end

    subgraph "TESTING ENVIRONMENT"
        TEST1[Unit Testing<br/>pytest Framework]
        TEST2[Integration Testing<br/>End-to-end Validation]
        TEST3[Security Testing<br/>Vulnerability Assessment]
        TEST4[Performance Testing<br/>Load & Stress Tests]
    end

    subgraph "BLOCKCHAIN INFRASTRUCTURE"
        BC1[Ganache<br/>Local Blockchain]
        BC2[Ethereum Testnet<br/>Ropsten/Goerli]
        BC3[Ethereum Mainnet<br/>Production Network]
        BC4[Smart Contracts<br/>Solidity Deployment]
    end

    subgraph "APPLICATION DEPLOYMENT"
        APP1[Streamlit Server<br/>Web Application]
        APP2[Python Backend<br/>Core Processing]
        APP3[Database Layer<br/>Configuration Storage]
        APP4[File System<br/>Local Storage]
    end

    subgraph "MONITORING & LOGGING"
        MON1[Application Logs<br/>System Monitoring]
        MON2[Performance Metrics<br/>Resource Usage]
        MON3[Security Logs<br/>Threat Detection]
        MON4[Error Tracking<br/>Issue Management]
    end

    subgraph "SECURITY INFRASTRUCTURE"
        SEC1[Firewall<br/>Network Protection]
        SEC2[SSL/TLS<br/>Encryption in Transit]
        SEC3[Key Management<br/>Secure Storage]
        SEC4[Access Control<br/>Authentication]
    end

    DEV1 --> TEST1
    DEV2 --> TEST2
    DEV3 --> TEST3
    DEV4 --> TEST4

    TEST1 --> BC1
    TEST2 --> BC2
    TEST3 --> BC3
    TEST4 --> BC4

    BC1 --> APP1
    BC2 --> APP2
    BC3 --> APP3
    BC4 --> APP4

    APP1 --> MON1
    APP2 --> MON2
    APP3 --> MON3
    APP4 --> MON4

    MON1 --> SEC1
    MON2 --> SEC2
    MON3 --> SEC3
    MON4 --> SEC4

    style DEV1 fill:#e8f5e8
    style TEST1 fill:#fff3e0
    style BC1 fill:#e3f2fd
    style APP1 fill:#fce4ec
    style MON1 fill:#f3e5f5
    style SEC1 fill:#ffebee
```

```mermaid
graph LR
    subgraph "IoT Layer"
        A1[Temperature Sensors]
        A2[Humidity Sensors]
        A3[Motion Sensors]
        A4[Light Sensors]
        A5[Air Quality Sensors]
    end

    subgraph "Privacy Layer"
        B1[Differential Privacy<br/>ε=1.0]
        B2[K-Anonymity<br/>k=3]
        B3[AES-256-CBC<br/>Encryption]
    end

    subgraph "Blockchain Layer"
        C1[Smart Contract]
        C2[Transaction Pool]
        C3[Block Mining]
        C4[Immutable Storage]
    end

    subgraph "ML Layer"
        D1[Data Buffer]
        D2[Feature Extraction]
        D3[Isolation Forest]
        D4[Anomaly Detection]
    end

    subgraph "Dashboard Layer"
        E1[Real-time Monitoring]
        E2[Blockchain Analytics]
        E3[File Management]
        E4[System Settings]
    end

    A1 --> B1
    A2 --> B2
    A3 --> B3
    A4 --> B1
    A5 --> B2

    B1 --> C1
    B2 --> C2
    B3 --> C3

    C1 --> D1
    C2 --> D2
    C3 --> D3
    C4 --> D4

    D1 --> E1
    D2 --> E2
    D3 --> E3
    D4 --> E4
```

## Privacy Pipeline Detailed Flow

```mermaid
sequenceDiagram
    participant S as IoT Sensor
    participant PP as Privacy Pipeline
    participant BC as Blockchain
    participant ML as ML Engine
    participant UI as Dashboard

    S->>PP: Raw Sensor Data
    PP->>PP: Apply Differential Privacy
    PP->>PP: Apply K-Anonymity
    PP->>PP: AES Encryption
    PP->>BC: Privacy-Preserved Data
    BC->>BC: Smart Contract Processing
    BC->>BC: Transaction Creation
    BC->>ML: Stored Data for Analysis
    ML->>ML: Check Data Threshold (≥10)
    ML->>ML: Auto-Train Model
    ML->>ML: Anomaly Detection
    ML->>UI: Results & Alerts
    UI->>UI: Real-time Visualization
```