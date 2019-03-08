import java.util.ArrayList;

/**
 * Class World - main data class.
 * 
 * Creates the world and manages the all data related to the locations,
 * connections between locations, the current location of the user, the current
 * orientation of the user (i.e., the direction the user is facing), the pick-up
 * items available at the current location, and the user’s inventory items. When
 * the user initiates the Forward command, the current location in the world is
 * changed while the orientation remains the same. Conversely, if the user
 * initiates the Left or Right commands, the orientation is updated while the
 * current location remains the same.
 * 
 * Orientation is modeled as a cyclical variable from 0 to 3, where each
 * increase in 1 modulo 4 (e.g., 0 to 1, or 3 to 0) represents a clockwise turn.
 * Orientation is used in conjunction with Location methods to establish
 * movement between locations and to identify the main image to display for the
 * current location.
 * 
 * @author Chris Sipola (s1667278)
 * @version 2016.11.25
 */
public class World {

	private Location currentLocation;
	private int orientation;
	private ArrayList<Item> inventory = new ArrayList<>();
	private ArrayList<Item> pickupItems = new ArrayList<>();

	/**
	 * Create logical map for world.
	 */
	public World() {

		Item math, sketch, jacket, coffee, floss;

		floss = new Item("floss", "floss2.jpg");
		// Source: http://www.kidspellinggames.com/dentist-spell-games.html

		math = new Item("impossibly hard math textbook", "math.jpg");
		// Source:
		// https://www.amazon.com/Algebraic-Topology-Allen-Hatcher/dp/0521795400

		sketch = new Item("architecture sketch", "sketch.jpg");
		// Source: https://www.pinterest.com/leahahaa/architecture/

		jacket = new Item("jacket", "jacket.png");
		// Source:
		// https://pixabay.com/en/jacket-winter-clothing-cold-wear-32714/

		coffee = new Item("caffeine", "coffee.png");
		// Source: https://pixabay.com/en/photos/coffee/

		Location myBedroomW, myBedroomE, myBathroom;
		Location hallway1, hallway2, davidsRoom;
		Location hallway3, hallway4, commonRoom;
		Location shonasRoomE, shonasRoomW;
		myBedroomW = new Location("myBedroomW", "my bedroom, west side");
		myBedroomE = new Location("myBedroomE", "my bedroom, east side");
		myBathroom = new Location("myBathroom", "my bathroom");
		hallway1 = new Location("hallway1", "hallway in front of room");
		hallway2 = new Location("hallway2", "hallway just north of room");
		davidsRoom = new Location("davidsRoom", "David's room");
		shonasRoomE = new Location("shonasRoomE", "Shona's bedroom, east side");
		shonasRoomW = new Location("shonasRoomW", "Shona's bedroom, west side");
		hallway3 = new Location("hallway3", "hallway, bending around corner");
		hallway4 = new Location("hallway4", "hallway, outside of common room");
		commonRoom = new Location("commonRoom", "common room");
		linkTwoLocations(myBedroomE, myBedroomW, 3);
		linkTwoLocations(myBedroomW, myBathroom, 2);
		linkTwoLocations(myBedroomW, hallway1, 3);
		linkTwoLocations(hallway1, hallway2, 0);
		linkTwoLocations(hallway2, davidsRoom, 1);
		linkTwoLocations(hallway2, shonasRoomE, 3);
		linkTwoLocations(shonasRoomE, shonasRoomW, 3);
		linkTwoLocations(hallway2, hallway3, 0);
		linkTwoLocations(hallway3, hallway4, 0);
		linkTwoLocations(hallway4, commonRoom, 3);

		myBedroomW.addItem(jacket);
		myBathroom.addItem(floss);
		davidsRoom.addItem(math);
		shonasRoomW.addItem(sketch);
		commonRoom.addItem(coffee);

		goToLocation(myBedroomE); // set starting location
		orientation = 0; // set starting orientation
	}

	/**
	 * Updates the world in response to a command.
	 * 
	 * @param command
	 *            a move command
	 */
	public void updateFromMoveCommand(MoveCommand command) {
		if (command == MoveCommand.FORWARD) {
			// Update location (and available pick-up items) but keep
			// orientation.
			moveForward();
		} else {
			// Update orientation but keep location.
			updateOrientation(command);
		}
	}

	/**
	 * Connect one location with another location by direction.
	 * 
	 * @param location1
	 *            first location
	 * @param location2
	 *            second location
	 * @param direction
	 *            direction pointing from first location to second location
	 */
	private void linkTwoLocations(Location location1, Location location2, int direction) {
		int backwardsDirection = (direction + 2) % 4;
		location1.connectLocations(direction, location2);
		location2.connectLocations(backwardsDirection, location1);
	}

	/**
	 * Go to location (and update pick-up items).
	 * 
	 * @param location
	 *            location to go to
	 */
	private void goToLocation(Location location) {
		currentLocation = location;
		pickupItems = currentLocation.getItems();
	}

	/**
	 * Get current location.
	 * 
	 * @return current location
	 */
	public Location getCurrentLocation() {
		return currentLocation;
	}

	/**
	 * Update orientation (turn left or right) given command.
	 * 
	 * @param command
	 *            move command
	 */
	public void updateOrientation(MoveCommand command) {

		switch (command) {
		case RIGHT:
			orientation = (orientation + 1) % 4; // clockwise
			break;
		case LEFT:
			orientation = (orientation + 3) % 4; // counterclockwise
			break;
		default:
			break;
		}
	}

	/**
	 * Get orientation direction.
	 * 
	 * @return orientation direction
	 */
	public int getOrientation() {
		return orientation;
	}

	/**
	 * Can you move forward given your location and orientation?
	 * 
	 * @return whether you can move forward
	 */
	public boolean canMoveForward() {
		return !(currentLocation.getConnectedLocation(orientation) == null);
	}

	/**
	 * Move forward to next location.
	 */
	public void moveForward() {
		goToLocation(currentLocation.getConnectedLocation(orientation));
	}

	/**
	 * Get inventory items.
	 * 
	 * @return inventory items
	 */
	public ArrayList<Item> getInventory() {
		return inventory;
	}

	/**
	 * Get pick-up items.
	 * 
	 * @return pick-up items
	 */
	public ArrayList<Item> getPickupItems() {
		return pickupItems;
	}
}
