import java.util.ArrayList;
import java.util.Iterator;
import java.io.Serializable;

public abstract class BLESchedule implements Serializable {

	protected int nodeID; // unique id for the device
	protected double beaconLength; // beacon length in milliseconds
	protected double simulationTime; // total time of simulation (i.e., of schedule) in milliseconds
	protected double startOffset = -1;

	protected BLEDiscSimulatorOptions options;

	protected ArrayList<BLEScheduleEvent> schedule = null;

	protected BLESchedule() {
	}

	public BLESchedule(int nodeID, BLEDiscSimulatorOptions options, double simulationTime) {
		this.nodeID = nodeID;
		this.simulationTime = simulationTime;
		this.options = options;
		this.beaconLength = options.getB();
	}

	public ArrayList<BLEScheduleEvent> getSchedule() {
		createSchedule();
		return schedule;
	}

	public ArrayList<BLEScheduleEvent> getEvents() {
		return schedule;
	}

	public double getStartOffset() {
		return startOffset;
	}

	abstract void createSchedule();

	public abstract void onDiscovery(BLEExtendedAdvertiseEndEvent base, BLEDiscSimulator simulation);

	public void printSchedule() {
		schedule.forEach(event -> System.out.println(event.toString()));
	}

	public double getDutyCycle() {
		double numerator = 0;
		Iterator<BLEScheduleEvent> it = schedule.iterator();
		while (it.hasNext()) {
			BLEScheduleEvent startEvent = it.next();
			BLEScheduleEvent endEvent = it.next();
			double startTime = startEvent.getTime();
			double endTime = endEvent.getTime();
			double elapsedTime = endTime - startTime;
			if (startEvent.isActivated()) {
				if (startEvent.isBeacon()) {
					numerator += elapsedTime * 3; // TODO: the *3 should really be dependent on how we're modeling the beacons BAD HACKY, works only when the beacon is 1.07ms in the settings file
				} else {
					numerator += elapsedTime;
				}
			}
		}
		return (numerator / (double) (simulationTime));
	}

	public double getConsumption() {
		double numerator = 0;
		double TXcost = options.getTXCost();
		double RXcost = options.getRXCost();
		double WPScanCost = options.getWpScanCost();
		double WPAdvCost = options.getWpAdvCost();
		Iterator<BLEScheduleEvent> it = schedule.iterator();
		while (it.hasNext()) {
			BLEScheduleEvent startEvent = it.next();
			BLEScheduleEvent endEvent = it.next();
			double startTime = startEvent.getTime();
			double endTime = endEvent.getTime();
			double elapsedTime = endTime - startTime;
			if (startEvent.isActivated()) {
				if (startEvent.isInWPScan()) {
					numerator += elapsedTime * WPScanCost;
				} else if (startEvent.isInWPAdv()) {
					numerator += elapsedTime * WPAdvCost;
				} else if (startEvent.isBeacon()) {
					numerator += elapsedTime * TXcost * 3; // TODO: the *3 should really be dependent on how we're modeling the beacons
				} else {
					numerator += elapsedTime * RXcost;
				}
			}
		}
		// numerator has the amount of energy consumed
		// so we're returning the instantaneous current (I in the optimizer)
		return (numerator / (double) (simulationTime));
	}

	public double getConsumptionWithIdleCost(double idleCost) {
		double numerator = 0;
		double TXcost = options.getTXCost();
		double RXcost = options.getRXCost();
		double radioOnTime = 0;
		double WPScanCost = options.getWpScanCost();
		double WPAdvCost = options.getWpAdvCost();
		double secondB = options.getSecondB();
		double secondaryChannelCost = options.getSecAdvCost(); //
		Iterator<BLEScheduleEvent> it = schedule.iterator();
		while (it.hasNext()) {
			BLEScheduleEvent startEvent = it.next();
			BLEScheduleEvent endEvent = it.next();
			double startTime = startEvent.getTime();
			double endTime = endEvent.getTime();
			double elapsedTime = endTime - startTime;
			radioOnTime += elapsedTime;
			if (startEvent.isActivated()) {
				if (startEvent.isInWPScan()) {
					numerator += elapsedTime * WPScanCost;
				} else if (startEvent.isInWPAdv()) {
					numerator += elapsedTime * WPAdvCost;
				} else if (startEvent.isBeacon() && secondB == 0) {
					numerator += elapsedTime * TXcost * 3; // TODO: the *3 should really be dependent on how we're modeling the beacons
				} else if (startEvent.isBeacon() && secondB > 0) {
					// System.out.print("Second");
					double AUX_Offset = options.getAUXOffset();
					double true_elapsed_time = 1 - AUX_Offset / elapsedTime;
					double ratio = beaconLength / secondB;
					numerator += true_elapsed_time * ratio * TXcost
							+ true_elapsed_time * (1 - ratio) * secondaryChannelCost;
					radioOnTime -= AUX_Offset;
				} else {
					numerator += elapsedTime * RXcost; // we consider that the secondary channel also use the 1M phy so the power consumption is the same as primary channel.
				}
			}
		}
		double idleTime = simulationTime - radioOnTime;
		numerator += idleTime * idleCost;
		// numerator has the amount of energy consumed
		// so we're returning the instantaneous current (I in the optimizer)
		return (numerator / (double) (simulationTime));
	}
	
	public int getNodeID() {
		return nodeID;
	}

	protected BLEListenStartEvent getNextListenStartEvent(double time) {
		if (schedule != null) {
			int i = 0;
			// skip all of the events that already happened (i.e., before time)
			while (i < schedule.size() && schedule.get(i).getTime() < time) {
				i++;
			}

			// now we need to look for events that belong to the node indicated by nodeID and are
			// BLEListenStartEvents
			while (i < schedule.size()) {
				BLEScheduleEvent bse = schedule.get(i);
				// Ugh. instanceof. Really, Christine? You can do better...
				if (bse instanceof BLEListenStartEvent) {
					return ((BLEListenStartEvent) bse);
				}
				i++;
			}
		}
		// I should only get here if a later listen start event for the node id doesn't exist
		return null;
	}

	protected BLEListenEndEvent getNextListenEndEvent(double time) {
		if (schedule != null) {
			int i = 0;
			// skip all of the events that already happened (i.e., before time)
			while (i < schedule.size() && schedule.get(i).getTime() < time) {
				i++;
			}
			// now we need to look for events that belong to the node indicated by nodeID and are
			// BLEListenEndEvents
			while (i < schedule.size()) {
				BLEScheduleEvent bse = schedule.get(i);
				if (bse instanceof BLEListenEndEvent) {
					return ((BLEListenEndEvent) bse);
				}
				i++;
			}
		}
		// I should only get here if a later listen start event for the node id doesn't exist
		return null;
	}

	protected BLEExtendedAdvertiseStartEvent getNextAdvertiseStartEvent(double time) {
		if (schedule != null) {
			int i = 0;
			// skip all of the events that already happened (i.e., before time)
			while (i < schedule.size() && schedule.get(i).getTime() < time) {
				i++;
			}
			// now we need to look for events that belong to the node indicated by nodeID and are
			// BLEExtendedAdvertiseStartEvents
			while (i < schedule.size()) {
				BLEScheduleEvent bse = schedule.get(i);
				if (bse instanceof BLEExtendedAdvertiseStartEvent) {
					return ((BLEExtendedAdvertiseStartEvent) bse);
				}
				i++;
			}
		}
		// I should only get here if a later listen start event  doesn't exist
		return null;
	}

	protected BLEExtendedAdvertiseEndEvent getNextAdvertiseEndEvent(double time) {
		if (schedule != null) {
			int i = 0;
			// skip all of the events that already happened (i.e., before time)
			while (i < schedule.size() && schedule.get(i).getTime() < time) {
				i++;
			}
			// now we need to look for events that belong to the node indicated by nodeID and are
			// BLEExtendedAdvertiseEndEvents
			while (i < schedule.size()) {
				BLEScheduleEvent bse = schedule.get(i);
				if (bse instanceof BLEExtendedAdvertiseEndEvent) {
					return ((BLEExtendedAdvertiseEndEvent) bse);
				}
				i++;
			}
		}
		// I should only get here if a later listen start event doesn't exist
		return null;
	}

}
