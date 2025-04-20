import java.util.ArrayList;
import java.util.Iterator;
import java.util.Random;

public class ExtendedSchedule extends BLESchedule{

    private double T; // epoch length in milliseconds
    private double L; // listen interval length in milliseconds
    private double advertisingInterval; // interval between beacons, in milliseconds
	private boolean addExtraEpoch; // does extra epoch added for fixing WP issue

	double totalBeaconTime;

	// Extended Adv propertities
    private double secondB; // the beaconLength of offloading data on secondary channels
    private double AUX_Offset; // the time between the beacons on primary channel and secondary channel
    
    private ExtendedSchedule(){}

    public ExtendedSchedule(int nodeID, BLEDiscSimulatorOptions options, double simulationTime, double[] startOffsets){
		super(nodeID, options, simulationTime);
		
		this.T = options.getT();
		this.addExtraEpoch = options.getAddExtra();
		this.secondB = options.getSecondB();
		this.AUX_Offset = options.getAUXOffset();

		// this.totalBeaconTime = threeB + AUX_Offset + secondB;
		this.totalBeaconTime = beaconLength + AUX_Offset + secondB;
		
		// the listening time is the time specified in the properties, then plus (s + AUX_offset + secondB)
		this.L = options.getL(); // It is the size of A
		L = L + options.getMaxAdditionalAdvDelay() + totalBeaconTime;

		this.advertisingInterval = options.getL();
		
		if(advertisingInterval < totalBeaconTime){
			System.err.println("Whoops! Invalid setting! The advertisement interval has to be longer than a beacon. This is probably the fault of the correction for BLE's added random advertising delay");
			System.exit(0);
		}
		setStartOffset(startOffsets, options, 1.5*T);
    }

    private void setStartOffset(double[] startOffsets, BLEDiscSimulatorOptions options, double range){
		// just select random number between 0 and  for the schedule's offset
		startOffset = (Math.random() * range);
    }

    // a schedule has to be three epochs long to represent
    // listening on the three channels in turn
    void createSchedule(){
		if(schedule == null){
			schedule = new ArrayList<BLEScheduleEvent>();
			double epochStartTime = startOffset;
			//int scanChannel = BLEScheduleEvent.ADVERTISEMENT_CHANNEL_ONE;
			// randomly choose one of the three scan channels to start on
			int scanChannel = (int) (Math.random() * 3);
			while(epochStartTime < simulationTime){
				double extraEpoch = 0;
				createOneEpoch(scanChannel, epochStartTime, extraEpoch);
				epochStartTime += (T+extraEpoch);
				scanChannel = BLEScheduleEvent.getNextScanChannel(scanChannel);
			}
		}
    }

    // this creates the schedule for one "epoch" which involves listening on only one channel
    private void createOneEpoch(int channel, double startTime, double extraEpoch){
		// listen for L milliseconds, starting at the provided startTime

		// Actual listening interval
		BLEListenStartEvent startListen = new BLEListenStartEvent(nodeID, startTime, channel);
		schedule.add(startListen);
		BLEListenEndEvent endListen = new BLEListenEndEvent(nodeID, startTime + L);
		schedule.add(endListen);

		// create the advertisement events and add them to the schedule
		double time = startTime + L; //advertising WP interval is before adv schedule
		double epoch = T+extraEpoch;

		double lastBeaconTime = time;

		// while(time+totalBeaconTime < (startTime + epoch - totalBeaconTime)){ //BLEnd
		while(time < (startTime + epoch - totalBeaconTime)){
			
			//secondary channel select
			int secondChannel = new Random().nextInt(37) + 3; // In our simulation, 3-39 as secondary channel and 0-2 as primary 
			BLEExtendedAdvertiseStartEvent startAdvertising = new BLEExtendedAdvertiseStartEvent(nodeID, time, secondChannel);
			schedule.add(startAdvertising);
			BLEExtendedAdvertiseEndEvent endAdvertising = new BLEExtendedAdvertiseEndEvent(nodeID, time+totalBeaconTime, secondChannel);
			schedule.add(endAdvertising);

			lastBeaconTime = time+totalBeaconTime;
			// right? the time between two start beacons should be the same as a listen (which is the advertising interval + beacon length)
			// the start time for the NEXT beacon should be time + advertisingInterval + the BLE random delay
			time = time + advertisingInterval;

			// select a random number between 0 and the max possible added random delay
			double randomDelay = Math.random() * options.getMaxAdditionalAdvDelay();
			time = time + randomDelay;
		}

		if(lastBeaconTime+totalBeaconTime<startTime+epoch){ //BLEnd
			// add last beacon
			int secondChannel = new Random().nextInt(37) + 3;
			BLEExtendedAdvertiseStartEvent startAdvertisingLast = new BLEExtendedAdvertiseStartEvent(nodeID, startTime+epoch-totalBeaconTime, secondChannel);
			schedule.add(startAdvertisingLast);
			BLEExtendedAdvertiseEndEvent endAdvertisingLast = new BLEExtendedAdvertiseEndEvent(nodeID, startTime+epoch, secondChannel);
			schedule.add(endAdvertisingLast);
		}
    }

    public void onDiscovery(BLEExtendedAdvertiseEndEvent base, BLEDiscSimulator simulation){
		// we only activate beacons if we're using the BLEnd half epoch model (which is the usual case)
    }
}
