import java.util.ArrayList;
import java.util.Iterator;
import java.util.Random;

public class NihaoExtendedAESchedule extends BLESchedule {

	private double E; // epoch length in milliseconds
	private double slotSize;

	private double totalBeaconTime;
	private double n;

	private double W;

	// Extended Adv propertities
	private double secondB; // the beaconLength of offloading data on secondary channels
	private double AUX_Offset; // the time between the beacons on primary channel and secondary channel

	private int CHUNKS; // total number of chunks
	private int chunkNumber = 0; // chunk index

	private NihaoExtendedAESchedule() {
	}

	public NihaoExtendedAESchedule(int nodeID, BLEDiscSimulatorOptions options, double simulationTime, double[] startOffsets) {
		super(nodeID, options, simulationTime);

		this.E = options.getT();
		this.n = options.getN();
		this.secondB = options.getSecondB();
		this.AUX_Offset = options.getAUXOffset();
		this.CHUNKS = options.getChunks();

		this.W = options.getW();

		this.totalBeaconTime = beaconLength + AUX_Offset + secondB;

		this.slotSize = E / n;

		if (slotSize < totalBeaconTime) {
			System.err.println(
					"Whoops! Invalid setting! The advertisement interval has to be longer than a beacon. This is probably the fault of the correction for BLE's added random advertising delay");
			System.exit(0);
		}
		setStartOffset(startOffsets, options, 2 * W);
	}

	// HC: CHANGING!
	private void setStartOffset(double[] startOffsets, BLEDiscSimulatorOptions options, double range) {
		// just select random number between 0 and  for the schedule's offset
		startOffset = (Math.random() * range);
	}

	void createSchedule() {
		if (schedule == null) {
			schedule = new ArrayList<BLEScheduleEvent>();
			double windowStartTime = startOffset;
			// randomly choose one of the three scan channels to start on
			int scanChannel = (int) (Math.random() * 3);

			while (windowStartTime < simulationTime) {
				double windowEndTime = windowStartTime + (W + W * Math.random()); //ExtendedAE
				double epochStartTime = windowStartTime;

				while (epochStartTime < windowStartTime + W) { // the rest of T does not adv or listen
					double epochEndTime = Math.min(windowStartTime + W, epochStartTime + E); 
					
					createOneEpoch(scanChannel, epochStartTime, 0, chunkNumber, epochEndTime);
					epochStartTime = epochEndTime;
					scanChannel = BLEScheduleEvent.getNextScanChannel(scanChannel);
				}

				chunkNumber = (int) (Math.random() * 1000000); // CHANGE
				windowStartTime = windowEndTime;
			}

		}
	}
	
	// this creates the schedule for one "epoch" which involves listening on only one channel
	private void createOneEpoch(int channel, double startTime, double extraEpoch, int chunkNumber, double epochEndTime) {

		if (startTime + totalBeaconTime <= epochEndTime) {
			int secondChannel = new Random().nextInt(37) + 3; // In our simulation, 3-39 as secondary channel and 0-2 as primary 

			BLEExtendedAdvertiseStartEvent startAdvertising = new BLEExtendedAdvertiseStartEvent(nodeID, startTime, secondChannel, chunkNumber);
			schedule.add(startAdvertising);
			BLEExtendedAdvertiseEndEvent endAdvertising = new BLEExtendedAdvertiseEndEvent(nodeID, startTime + totalBeaconTime, secondChannel, chunkNumber);
			schedule.add(endAdvertising);
		}
		
		if (startTime + totalBeaconTime >= epochEndTime) {
			return;
		}

		// Actual listening interval
		BLEListenStartEvent startListen = new BLEListenStartEvent(nodeID, startTime+totalBeaconTime, channel);
		schedule.add(startListen);
		
		double endListenTime = Math.min(startTime + slotSize, epochEndTime);
		BLEListenEndEvent endListen = new BLEListenEndEvent(nodeID, endListenTime);
		schedule.add(endListen);

		// create the advertisement events and add them to the schedule
		double time = endListenTime;

		while (time + totalBeaconTime <= epochEndTime) {
			int secondChannel = new Random().nextInt(37) + 3; // In our simulation, 3-39 as secondary channel and 0-2 as primary 

			BLEExtendedAdvertiseStartEvent startAdvertising = new BLEExtendedAdvertiseStartEvent(nodeID, time, secondChannel, chunkNumber);
			schedule.add(startAdvertising);
			BLEExtendedAdvertiseEndEvent endAdvertising = new BLEExtendedAdvertiseEndEvent(nodeID, time+totalBeaconTime, secondChannel, chunkNumber);
			schedule.add(endAdvertising);

			time += slotSize;
		}

	}
	

	public void onDiscovery(BLEExtendedAdvertiseEndEvent base, BLEDiscSimulator simulation) {
		// we only activate beacons if we're using the BLEnd half epoch model (which is the usual case)
	}
}