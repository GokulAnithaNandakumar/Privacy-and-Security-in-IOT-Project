// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title IoTDataStorage
 * @dev Smart contract for secure IoT sensor data storage with privacy preservation
 * @notice Stores encrypted IoT sensor data on blockchain for tamper-proof audit trails
 */
contract IoTDataStorage {

    // Structure to store IoT sensor data
    struct SensorData {
        string sensorId;           // Unique sensor identifier
        string room;               // Room location
        string sensorType;         // Type of sensor (temperature, motion, etc.)
        bytes32 encryptedValue;    // Encrypted sensor value (privacy preserved)
        bytes32 dataHash;          // Hash of original data for integrity
        uint256 timestamp;         // Block timestamp
        address recorder;          // Address that recorded the data
        bool isAnomalous;         // ML intrusion detection result
        uint256 confidenceScore;  // ML confidence score (0-100)
    }

    // Structure for privacy metadata
    struct PrivacyMetadata {
        uint256 differentialPrivacyEpsilon;  // ε value for differential privacy
        uint256 kAnonymityLevel;             // K-anonymity level
        string encryptionAlgorithm;          // Encryption method used
    }

    // Events for real-time monitoring
    event SensorDataRecorded(
        uint256 indexed dataId,
        string indexed sensorId,
        string indexed room,
        uint256 timestamp,
        bool isAnomalous
    );

    event IntrusionDetected(
        uint256 indexed dataId,
        string indexed sensorId,
        uint256 confidenceScore,
        uint256 timestamp
    );

    event PrivacyPolicyUpdated(
        uint256 epsilon,
        uint256 kLevel,
        string algorithm
    );

    // File storage structure
    struct FileData {
        string fileName;           // Original file name
        bytes32 fileHash;         // Hash of file content
        bytes encryptedContent;   // Encrypted file content (for small files)
        string ipfsHash;          // IPFS hash for large files
        uint256 fileSize;         // File size in bytes
        string contentType;       // MIME type
        uint256 timestamp;        // Upload timestamp
        address uploader;         // Address that uploaded the file
        bool isPublic;           // Public or private file
    }

    // File events
    event FileStored(
        uint256 indexed fileId,
        string indexed fileName,
        bytes32 fileHash,
        uint256 fileSize,
        address indexed uploader,
        uint256 timestamp
    );

    event FileAccessed(
        uint256 indexed fileId,
        address indexed accessor,
        uint256 timestamp
    );

    // State variables
    mapping(uint256 => SensorData) public sensorDataRecords;
    mapping(uint256 => FileData) public fileRecords;
    mapping(address => uint256[]) public userFiles;  // user => fileIds
    mapping(string => uint256[]) public sensorHistory;  // sensorId => dataIds
    mapping(string => uint256[]) public roomData;       // room => dataIds

    PrivacyMetadata public privacyPolicy;

    uint256 public dataCounter;
    uint256 public fileCounter;
    uint256 public anomalousDataCount;

    address public owner;
    mapping(address => bool) public authorizedRecorders;

    // Modifiers
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can perform this action");
        _;
    }

    modifier onlyAuthorized() {
        require(authorizedRecorders[msg.sender] || msg.sender == owner,
                "Only authorized recorders can add data");
        _;
    }

    /**
     * @dev Constructor sets the deployer as owner and initial privacy policy
     */
    constructor() {
        owner = msg.sender;
        authorizedRecorders[msg.sender] = true;
        dataCounter = 0;
        fileCounter = 0;
        anomalousDataCount = 0;

        // Set default privacy policy
        privacyPolicy = PrivacyMetadata({
            differentialPrivacyEpsilon: 100,  // ε = 1.0 (scaled by 100)
            kAnonymityLevel: 3,
            encryptionAlgorithm: "AES-256-CBC"
        });
    }

    /**
     * @dev Add authorized recorder address
     */
    function addAuthorizedRecorder(address recorder) external onlyOwner {
        authorizedRecorders[recorder] = true;
    }

    /**
     * @dev Remove authorized recorder
     */
    function removeAuthorizedRecorder(address recorder) external onlyOwner {
        authorizedRecorders[recorder] = false;
    }

    /**
     * @dev Update privacy policy parameters
     */
    function updatePrivacyPolicy(
        uint256 _epsilon,
        uint256 _kLevel,
        string memory _algorithm
    ) external onlyOwner {
        privacyPolicy.differentialPrivacyEpsilon = _epsilon;
        privacyPolicy.kAnonymityLevel = _kLevel;
        privacyPolicy.encryptionAlgorithm = _algorithm;

        emit PrivacyPolicyUpdated(_epsilon, _kLevel, _algorithm);
    }

    /**
     * @dev Record IoT sensor data on blockchain
     * @param _sensorId Unique identifier for the sensor
     * @param _room Room where sensor is located
     * @param _sensorType Type of sensor
     * @param _encryptedValue Privacy-preserved encrypted value
     * @param _dataHash Hash of original data for integrity verification
     * @param _isAnomalous Result from ML intrusion detection
     * @param _confidenceScore ML confidence score (0-100)
     */
    function recordSensorData(
        string memory _sensorId,
        string memory _room,
        string memory _sensorType,
        bytes32 _encryptedValue,
        bytes32 _dataHash,
        bool _isAnomalous,
        uint256 _confidenceScore
    ) external onlyAuthorized returns (uint256) {

        require(bytes(_sensorId).length > 0, "Sensor ID cannot be empty");
        require(bytes(_room).length > 0, "Room cannot be empty");
        require(bytes(_sensorType).length > 0, "Sensor type cannot be empty");
        require(_confidenceScore <= 100, "Confidence score must be 0-100");

        dataCounter++;

        // Create sensor data record
        sensorDataRecords[dataCounter] = SensorData({
            sensorId: _sensorId,
            room: _room,
            sensorType: _sensorType,
            encryptedValue: _encryptedValue,
            dataHash: _dataHash,
            timestamp: block.timestamp,
            recorder: msg.sender,
            isAnomalous: _isAnomalous,
            confidenceScore: _confidenceScore
        });

        // Update indexes
        sensorHistory[_sensorId].push(dataCounter);
        roomData[_room].push(dataCounter);

        // Track anomalies
        if (_isAnomalous) {
            anomalousDataCount++;
            emit IntrusionDetected(dataCounter, _sensorId, _confidenceScore, block.timestamp);
        }

        emit SensorDataRecorded(dataCounter, _sensorId, _room, block.timestamp, _isAnomalous);

        return dataCounter;
    }

    /**
     * @dev Get sensor data by ID
     */
    function getSensorData(uint256 _dataId) external view returns (
        string memory sensorId,
        string memory room,
        string memory sensorType,
        bytes32 encryptedValue,
        bytes32 dataHash,
        uint256 timestamp,
        address recorder,
        bool isAnomalous,
        uint256 confidenceScore
    ) {
        require(_dataId <= dataCounter && _dataId > 0, "Invalid data ID");

        SensorData memory data = sensorDataRecords[_dataId];
        return (
            data.sensorId,
            data.room,
            data.sensorType,
            data.encryptedValue,
            data.dataHash,
            data.timestamp,
            data.recorder,
            data.isAnomalous,
            data.confidenceScore
        );
    }

    /**
     * @dev Get sensor history for a specific sensor
     */
    function getSensorHistory(string memory _sensorId) external view returns (uint256[] memory) {
        return sensorHistory[_sensorId];
    }

    /**
     * @dev Get all data for a specific room
     */
    function getRoomData(string memory _room) external view returns (uint256[] memory) {
        return roomData[_room];
    }

    /**
     * @dev Get contract statistics
     */
    function getContractStats() external view returns (
        uint256 totalRecords,
        uint256 anomalousRecords,
        uint256 currentEpsilon,
        uint256 kLevel,
        string memory encryptionMethod
    ) {
        return (
            dataCounter,
            anomalousDataCount,
            privacyPolicy.differentialPrivacyEpsilon,
            privacyPolicy.kAnonymityLevel,
            privacyPolicy.encryptionAlgorithm
        );
    }

    /**
     * @dev Get recent anomalous data (last N records)
     */
    function getRecentAnomalies(uint256 _limit) external view returns (uint256[] memory) {
        uint256[] memory anomalies = new uint256[](_limit);
        uint256 found = 0;

        // Search backwards from latest data
        for (uint256 i = dataCounter; i > 0 && found < _limit; i--) {
            if (sensorDataRecords[i].isAnomalous) {
                anomalies[found] = i;
                found++;
            }
        }

        // Resize array to actual found count
        uint256[] memory result = new uint256[](found);
        for (uint256 j = 0; j < found; j++) {
            result[j] = anomalies[j];
        }

        return result;
    }

    /**
     * @dev Store file on blockchain with privacy preservation
     */
    function storeFile(
        string memory _fileName,
        bytes32 _fileHash,
        bytes memory _encryptedContent,
        string memory _ipfsHash,
        uint256 _fileSize,
        string memory _contentType,
        bool _isPublic
    ) external onlyAuthorized returns (uint256) {
        fileCounter++;
        uint256 fileId = fileCounter;

        fileRecords[fileId] = FileData({
            fileName: _fileName,
            fileHash: _fileHash,
            encryptedContent: _encryptedContent,
            ipfsHash: _ipfsHash,
            fileSize: _fileSize,
            contentType: _contentType,
            timestamp: block.timestamp,
            uploader: msg.sender,
            isPublic: _isPublic
        });

        userFiles[msg.sender].push(fileId);

        emit FileStored(
            fileId,
            _fileName,
            _fileHash,
            _fileSize,
            msg.sender,
            block.timestamp
        );

        return fileId;
    }

    /**
     * @dev Access file data (for authorized users)
     */
    function accessFile(uint256 _fileId) external returns (FileData memory) {
        require(_fileId > 0 && _fileId <= fileCounter, "File does not exist");

        FileData memory file = fileRecords[_fileId];

        // Check access permissions
        require(
            file.isPublic ||
            file.uploader == msg.sender ||
            msg.sender == owner,
            "Access denied"
        );

        emit FileAccessed(_fileId, msg.sender, block.timestamp);

        return file;
    }

    /**
     * @dev Get user's files
     */
    function getUserFiles(address _user) external view returns (uint256[] memory) {
        return userFiles[_user];
    }

    /**
     * @dev Get file info without content (public view)
     */
    function getFileInfo(uint256 _fileId) external view returns (
        string memory fileName,
        bytes32 fileHash,
        uint256 fileSize,
        string memory contentType,
        uint256 timestamp,
        address uploader,
        bool isPublic
    ) {
        require(_fileId > 0 && _fileId <= fileCounter, "File does not exist");

        FileData memory file = fileRecords[_fileId];

        return (
            file.fileName,
            file.fileHash,
            file.fileSize,
            file.contentType,
            file.timestamp,
            file.uploader,
            file.isPublic
        );
    }

    /**
     * @dev Get comprehensive contract statistics including files
     */
    function getFullContractStats() external view returns (
        uint256 totalRecords,
        uint256 totalFiles,
        uint256 anomalousRecords,
        uint256 currentEpsilon,
        uint256 kLevel,
        string memory encryptionMethod
    ) {
        return (
            dataCounter,
            fileCounter,
            anomalousDataCount,
            privacyPolicy.differentialPrivacyEpsilon,
            privacyPolicy.kAnonymityLevel,
            privacyPolicy.encryptionAlgorithm
        );
    }

    /**
     * @dev Verify data integrity using hash
     */
    function verifyDataIntegrity(uint256 _dataId, bytes32 _originalHash) external view returns (bool) {
        require(_dataId <= dataCounter && _dataId > 0, "Invalid data ID");
        return sensorDataRecords[_dataId].dataHash == _originalHash;
    }

    /**
     * @dev Emergency pause function (only owner)
     */
    bool public paused = false;

    modifier whenNotPaused() {
        require(!paused, "Contract is paused");
        _;
    }

    function pauseContract() external onlyOwner {
        paused = true;
    }

    function unpauseContract() external onlyOwner {
        paused = false;
    }
}