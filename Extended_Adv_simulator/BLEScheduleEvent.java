import java.io.Serializable;

public abstract class BLEScheduleEvent implements Serializable{

    public static int ADVERTISEMENT_CHANNEL_ONE = 0;
    public static int ADVERTISEMENT_CHANNEL_TWO = 1;
    public static int ADVERTISEMENT_CHANNEL_THREE = 2;
    
    double time; //microseconds since the start of the simulation
    int nodeID; // unique nodeID of the node initiating the event
    boolean start; // true if this is a "start" event; false if this is an "end" event

    protected boolean isActivated = true;
    protected boolean extra = false;
    protected boolean isPkLoss = false;
    protected boolean isWpScanInterval = false;
    protected boolean isWpAdvInterval = false;

    BLEScheduleEvent(){}
    
    public BLEScheduleEvent(int nodeID, double time, boolean start){
	this.time = time;
	this.nodeID = nodeID;
	this.start = start;
    }

    public double getTime(){
	return time;
    }

    public int getNodeID(){
	return nodeID;
    }

    public boolean getStart(){
	return start;
    }

    public static int getNextScanChannel(int currentScanChannel){
	int toReturn = currentScanChannel + 1;
	if(toReturn > 2){
	    toReturn = 0;
	}
	return toReturn;
    }
    
    public boolean isActivated(){
	return isActivated;
    }

    public void setActive(boolean active){
	this.isActivated = active;
    }

    public boolean isExtra(){
	return extra;
    }

    public void setExtra(boolean extra){
	this.extra = extra;
    }

    public boolean isPkLoss(){
    return isPkLoss;
    }

    public void setIsPkLoss(boolean isPkLoss){
    this.isPkLoss = isPkLoss;
    }

    public boolean isInWPScan(){
    return isWpScanInterval;
    }

    public boolean isInWPAdv(){
        return isWpAdvInterval;
    }
    
    public void setIsInWPScan(boolean isWPScan){
    this.isWpScanInterval = isWPScan;
    }

    public void setIsInWPAdv(boolean isWPAdv){
        this.isWpAdvInterval = isWPAdv;
    }

    public abstract void process(BLEDiscSimulator simulator);

    public abstract boolean isBeacon();


}
