import java.io.File;
import java.io.FileInputStream;
import java.util.Properties;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.Serializable;

public class BLEDiscSimulatorOptions implements Serializable{

    // if a random additional delay is added to the advertisement delay (as is done in BLE),
    // it is necessary to "correct" for it to achieve the BLEnd delivery guarantees.
    // there are three options: not to correct at all (and not achieve the guarantees),
    // to correct in the advertisement interval (by shortening the fixed advDelay), or
    // to correct in the listening interval (by extending the listening period).
    public static int ADV_DELAY_CORRECT_NONE = 0;
    public static int ADV_DELAY_CORRECT_ADVERTISE = 1;
    public static int ADV_DELAY_CORRECT_LISTEN = 2;

    public static int LOG_STYLE_BRIEF = 0;
    public static int LOG_STYLE_VERBOSE = 1;
    public static int LOG_STYLE_CDF = 2;

    public static int PROTOCOL_BLEND = 0;
    // public static int PROTOCOL_SEARCHLIGHT = 1;
    public static int PROTOCOL_NIHAO = 2;
	public static int METHOD_MIX = 3;
	public static int METHOD_EXTENDED = 4;
	public static int METHOD_EXTENDED_NIHAO = 5;

    private static int DEFAULT_NUM_NODES = 10;
    private static double DEFAULT_T = 1000;
    private static double DEFAULT_L = 100;
    private static double DEFAULT_SLOT_LENGTH = 100;
    private static double DEFAULT_SIMULATION_TIME = 3000;
	private static double DEFAULT_B = 1.1;
	private static double DEFAULT_SECOND_B = 2; // 1 Mbps for 255 Bytes needs 2.04ms
	private static double DEFAULT_AUX_OFFSET = 3; 
	private static double DEFAULT_M = 0;
    private static int DEFAULT_N = 10;
    private static double DEFAULT_ADDITIONAL_ADV_DELAY = 0;
    private static int DEFAULT_ADV_DELAY_CORRECT_TYPE = ADV_DELAY_CORRECT_NONE;
    private static boolean DEFAULT_BIDIRECTIONAL_DISCOVERY_ENABLED = false;
    private static boolean DEFAULT_LOAD_SCHEDULES_FROM_FILE = false;
    private static boolean DEFAULT_SAVE_SCHEDULES_TO_FILE = false;
    private static String DEFAULT_SCHEDULE_LOAD_FILE = "schedule";
    private static String DEFAULT_SCHEDULE_SAVE_FILE = "schedule";
    private static int DEFAULT_LOG_STYLE = LOG_STYLE_BRIEF;
    private static int DEFAULT_PROTOCOL = PROTOCOL_BLEND;
    private static boolean DEFAULT_MODEL_COLLISIONS = true;
    private static boolean DEFAULT_MODEL_CHANNELS = false;
    private static boolean DEFAULT_MODEL_BLEND_HALF_EPOCH = true;
    private static boolean DEFAULT_CONTROL_START_OFFSET = false;
    private static boolean DEFAULT_SHOW_SCHEDULES = false;
    private static boolean DEFAULT_PRINT_STATISTICS = false;
    //private static double DEFAULT_TX_COST = 5.725;
    //private static double DEFAULT_RX_COST = 6.329;
	// private static double DEFAULT_TX_COST = 3.578; // primary channel
	private static double DEFAULT_TX_COST = 2.64; // primary channel
	private static double DEFAULT_RX_COST = 2.108;
	private static double DEFAULT_WP_SCAN_COST = 0.406;
	private static double DEFAULT_WP_ADV_COST = 0.339;
	private static double  DEFAULT_SEC_B_COST = 7.98; // secondary channel

	private static boolean DEFAULT_ADD_EXTRA = true;
	private static double DEFAULT_WP_SCAN = 0;
	private static double DEFAULT_WP_ADV = 0;

	private static int DEFAULT_W = 0;

	private static int DEFAULT_CHUNK = 1; //multi-adv only

    private Properties properties = null;

    public BLEDiscSimulatorOptions(String propertiesFile){
	try{
	    File file = new File(propertiesFile);
	    FileInputStream fileInput = new FileInputStream(file);
	    properties = new Properties();
	    properties.load(fileInput);
	    fileInput.close();
	} catch(FileNotFoundException fnfe) {
	    fnfe.printStackTrace();
	} catch(IOException ioe) {
	    ioe.printStackTrace();
	}
    }

    public int getNumNodes(){
	String numNodes = null;
	if(properties != null){
	    numNodes = properties.getProperty("numNodes");
	}
	if(numNodes != null){
	    return Integer.parseInt(numNodes);
	}
	else{
	    return DEFAULT_NUM_NODES;
	}
    }

    public double getTXCost(){
	String TXCost = null;
	if(properties != null){
	    TXCost = properties.getProperty("TXCost");
	}
	if(TXCost != null){
	    return Double.parseDouble(TXCost);
	}
	else{
	    return DEFAULT_TX_COST;
	}
    }

    public double getRXCost(){
	String RXCost = null;
	if(properties != null){
	    RXCost = properties.getProperty("RXCost");
	}
	if(RXCost != null){
	    return Double.parseDouble(RXCost);
	}
	else{
	    return DEFAULT_RX_COST;
	}
    }

	public double getWpScanCost(){
		String WpScanCost = null;
		if(properties != null){
			WpScanCost = properties.getProperty("WpScanCost");
		}
		if(WpScanCost != null){
			return Double.parseDouble(WpScanCost);
		}
		else{
			return DEFAULT_WP_SCAN_COST;
		}
	}

	public double getWpAdvCost() {
		String WpAdvCost = null;
		if (properties != null) {
			WpAdvCost = properties.getProperty("WpAdvCost");
		}
		if (WpAdvCost != null) {
			return Double.parseDouble(WpAdvCost);
		} else {
			return DEFAULT_WP_ADV_COST;
		}
	}
	
	public double getSecAdvCost(){
		String secAdvCost = null;
		if(properties != null){
			secAdvCost = properties.getProperty("secAdvCost");
		}
		if(secAdvCost != null){
			return Double.parseDouble(secAdvCost);
		}
		else{
			return DEFAULT_SEC_B_COST;
		}
	}
    
    public double getT(){
	String T = null;
	if(properties != null){
	    T = properties.getProperty("T");
	}
	if(T != null){
	    return Double.parseDouble(T);
	}
	else{
	    return DEFAULT_T;
	}
    }

    public double getL(){
	String L = null;
	if(properties != null){
	    L = properties.getProperty("L");
	}
	if(L != null){
	    return Double.parseDouble(L);
	}
	else{
	    return DEFAULT_L;
	}
    }

    public double getSlotLength(){
	String slotLength = null;
	if(properties != null){
	    slotLength = properties.getProperty("slotLength");
	}
	if(slotLength != null){
	    return Double.parseDouble(slotLength);
	}
	else{
	    return DEFAULT_SLOT_LENGTH;
	}
    }

    public double getSimulationTime(){
	String simulationTime = null;
	if(properties != null){
	    simulationTime = properties.getProperty("simulationTime");
	}
	if(simulationTime != null){
	    return Double.parseDouble(simulationTime);
	}
	else{
	    return DEFAULT_SIMULATION_TIME;
	}
    }

	public double getM(){
	String m = null;
	if(properties != null){
	    m = properties.getProperty("m");
	}
	if(m != null){
	    return Double.parseDouble(m);
	}
	else{
	    return DEFAULT_M;
	}
    }

	public boolean getAddExtra(){
		String addExtra = null;
		if(properties != null){
			addExtra = properties.getProperty("addExtra");
		}
		if(addExtra != null){
			return Boolean.parseBoolean(addExtra);
		}
		else{
			return DEFAULT_ADD_EXTRA;
		}
	}

	public double getWP_Scan(){
	String wp = null;
	if(properties != null){
		wp = properties.getProperty("wp_scan");
	}
	if(wp != null){
		return Double.parseDouble(wp);
	}
	else{
		return DEFAULT_WP_SCAN;
	}
	}

	public double getWP_Adv(){
		String wp = null;
		if(properties != null){
			wp = properties.getProperty("wp_adv");
		}
		if(wp != null){
			return Double.parseDouble(wp);
		}
		else{
			return DEFAULT_WP_ADV;
		}
	}

    public double getB(){
	String b = null;
	if(properties != null){
	    b = properties.getProperty("b");
	}
	if(b != null){
	    return Double.parseDouble(b);
	}
	else{
	    return DEFAULT_B;
	}
    }

	public double getSecondB(){
		String b_second = null;
		if(properties != null){
			b_second = properties.getProperty("b_second");
		}
		if(b_second != null){
			return Double.parseDouble(b_second);
		}
		else{
			return DEFAULT_SECOND_B;
		}
	}

	public double getAUXOffset(){
		String AUX_Offset = null;
		if(properties != null){
			AUX_Offset = properties.getProperty("AUX_Offset");
		}
		if(AUX_Offset != null){
			return Double.parseDouble(AUX_Offset);
		}
		else{
			return DEFAULT_AUX_OFFSET;
		}
	}

	public int getN() {
		String n = null;
		if (properties != null) {
			n = properties.getProperty("n");
		}
		if (n != null) {
			return Integer.parseInt(n);
		} else {
			return DEFAULT_N;
		}
	}
	
	public int getW(){
		String W = null;
		if(properties != null){
			W = properties.getProperty("W");
		}
		if(W != null){
			return Integer.parseInt(W);
		}
		else{
			return DEFAULT_W;
		}
	}

	public int getChunks(){
		String chunk = null;
		if(properties != null){
			chunk = properties.getProperty("chunks");
		}
		if(chunk != null){
			return Integer.parseInt(chunk);
		}
		else{
			return DEFAULT_CHUNK;
		}
	}

    public double getMaxAdditionalAdvDelay(){
	String maxAdditionalAdvDelay = null;
	if(properties != null){
	    maxAdditionalAdvDelay = properties.getProperty("maxAdditionalAdvDelay");
	}
	if(maxAdditionalAdvDelay != null){
	    return Double.parseDouble(maxAdditionalAdvDelay);
	}
	else{
	    return DEFAULT_ADDITIONAL_ADV_DELAY;
	}
    }

    public int correctAdvDelay(){
	String correctAdvDelayType = null;
	if(properties != null){
	    correctAdvDelayType = properties.getProperty("correctAdvDelayType");
	}
	if(correctAdvDelayType != null){
	    if(correctAdvDelayType.equals("LISTEN")){
		return ADV_DELAY_CORRECT_LISTEN;
	    }
	    else if(correctAdvDelayType.equals("ADVERTISE")){
		return ADV_DELAY_CORRECT_ADVERTISE;
	    }
	    else{
		 return ADV_DELAY_CORRECT_NONE;
	    }
	}
	else{
	    return DEFAULT_ADV_DELAY_CORRECT_TYPE;
	}
    }

    public boolean controlStartOffset(){
	String controlStartOffset = null;
	if(properties != null){
	    controlStartOffset = properties.getProperty("controlStartOffset");
	}
	if(controlStartOffset != null){
	    return Boolean.valueOf(controlStartOffset);
	}
	else{
	    return DEFAULT_CONTROL_START_OFFSET;
	}
    }

    public boolean modelCollisions(){
	String modelCollisions = null;
	if(properties != null){
	    modelCollisions = properties.getProperty("modelCollisions");
	}
	if(modelCollisions != null){
	    return Boolean.valueOf(modelCollisions);
	}
	else{
	    return DEFAULT_MODEL_COLLISIONS;
	}
    }

    public boolean modelChannels(){
	String modelChannels = null;
	if(properties != null){
	    modelChannels = properties.getProperty("modelChannels");
	}
	if(modelChannels != null){
	    return Boolean.valueOf(modelChannels);
	}
	else{
	    return DEFAULT_MODEL_CHANNELS;
	}
    }

    public boolean modelBLEndHalfEpoch(){
	String modelBLEndHalfEpoch = null;
	if(properties != null){
	    modelBLEndHalfEpoch = properties.getProperty("modelBLEndHalfEpoch");
	}
	if(modelBLEndHalfEpoch != null){
	    return Boolean.valueOf(modelBLEndHalfEpoch);
	}
	else{
	    return DEFAULT_MODEL_BLEND_HALF_EPOCH;
	}
    }
    
    public boolean isBidirectionalDiscoveryEnabled(){
	String bidirectionalDiscoveryEnabled = null;
	if(properties != null){
	    bidirectionalDiscoveryEnabled = properties.getProperty("bidirectionalDiscoveryEnabled");
	}
	if(bidirectionalDiscoveryEnabled != null){
	    return Boolean.valueOf(bidirectionalDiscoveryEnabled);
	}
	else{
	    return DEFAULT_BIDIRECTIONAL_DISCOVERY_ENABLED;
	}
    }

    public boolean printStatistics(){
	String printStatistics = null;
	if(properties != null){
	    printStatistics = properties.getProperty("printStatistics");
	}
	if(printStatistics != null){
	    return  Boolean.valueOf(printStatistics);
	}
	else{
	    return DEFAULT_PRINT_STATISTICS;
	}
    }
    
    public boolean saveSchedulesToFile(){
	String saveSchedulesToFile = null;
	if(properties != null){
	    saveSchedulesToFile = properties.getProperty("saveSimulationSchedule");
	}
	if(saveSchedulesToFile != null){
	    return  Boolean.valueOf(saveSchedulesToFile);
	}
	else{
	    return DEFAULT_LOAD_SCHEDULES_FROM_FILE;
	}
    }


    public boolean loadSchedulesFromFile(){
	String loadSchedulesFromFile = null;
	if(properties != null){
	    loadSchedulesFromFile = properties.getProperty("loadSimulationSchedule");
	}
	if(loadSchedulesFromFile != null){
	    return  Boolean.valueOf(loadSchedulesFromFile);
	}
	else{
	    return DEFAULT_LOAD_SCHEDULES_FROM_FILE;
	}
    }

    public String getScheduleLoadFile(){
	String scheduleLoadFile = null;
	if(properties != null){
	    scheduleLoadFile = properties.getProperty("scheduleLoadFile");
	}
	if(scheduleLoadFile != null){
	    return scheduleLoadFile;
	}
	else{
	    return DEFAULT_SCHEDULE_LOAD_FILE;
	}
    }

    public String getScheduleSaveFile(){
	String scheduleSaveFile = null;
	if(properties != null){
	    scheduleSaveFile = properties.getProperty("scheduleSaveFile");
	}
	if(scheduleSaveFile != null){
	    return scheduleSaveFile;
	}
	else{
	    return DEFAULT_SCHEDULE_SAVE_FILE;
	}
    }

    public int getLogStyle(){
	String logStyle = null;
	if(properties != null){
	    logStyle = properties.getProperty("logStyle");
	}
	if(logStyle.equals("brief")){
	    return LOG_STYLE_BRIEF;
	}
	else if(logStyle.equals("verbose")){
	    return LOG_STYLE_VERBOSE;
	}
	else if(logStyle.equals("cdf")){
	    return LOG_STYLE_CDF;
	}
	else{
	    return DEFAULT_LOG_STYLE;
	}
    }

    public int getProtocol(){
	String protocol = null;
	if(properties != null){
	    protocol = properties.getProperty("protocol");
	}
	if(protocol.equals("blend")){
	    return PROTOCOL_BLEND;
	}
	// else if(protocol.equals("searchlight")){
	//     return PROTOCOL_SEARCHLIGHT;
	// }
	else if(protocol.equals("nihao")){
	    return PROTOCOL_NIHAO;
	}
	else if (protocol.equals("mix")) {
		return METHOD_MIX;
	}
	else if (protocol.equals("extended")) {
		return METHOD_EXTENDED;
	}
	else if(protocol.equals("extendednihao")){
	    return METHOD_EXTENDED_NIHAO;
	}
	else{
	    return DEFAULT_PROTOCOL;
	}
    }

    public boolean showSchedules(){
	String showSchedules = null;
	if(properties != null){
	    showSchedules = properties.getProperty("showSchedules");
	}
	if(showSchedules != null){
	    return  Boolean.valueOf(showSchedules);
	}
	else{
	    return DEFAULT_SHOW_SCHEDULES;
	}
    }

    public String toString(){
	return ("numNodes = " + getNumNodes() +
		"\nT = " + getT() +
		"\nL = " + getL() +
		"\nb = " + getB() +
		"\nMaxAddlDelay = " + getMaxAdditionalAdvDelay() +
		"\ncorrection style = " + correctAdvDelay() +
		"\nbidirectional discovery enabled = " + isBidirectionalDiscoveryEnabled() + "\n\n");
    }
	
}
