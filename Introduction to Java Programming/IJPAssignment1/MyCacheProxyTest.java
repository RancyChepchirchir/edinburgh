// IJP Assignment 1, Version 6.1.0, 05 Oct 2016
package ijp.test;

import static org.junit.Assert.*;
import java.awt.image.BufferedImage;

import org.junit.Before;
import org.junit.After;
import org.junit.Test;

import ijp.Picture;
import ijp.proxy.BrokenCacheProxy;
import ijp.service.Service;

/**
 * Test a cache proxy for the PictureViewer application. Code based on
 * given template.
 * 
 * @author Chris Sipola
 * @version 1.0
 */
public class MyCacheProxyTest implements Service {
	
	Service proxy;
	
	@Before
	public void setUp() {
		// set proxy to be used for all unit tests
		proxy = new BrokenCacheProxy(this);
	}
	
	@After
	public void tearDown() {
		// nothing
	}
	
	/**
	 * Test that requests for the same subject and index return the same image.
	 */
	@Test
	public void equalityTest() {
		
		Picture firstPicture = proxy.getPicture("equalityTest",2);
		Picture secondPicture = proxy.getPicture("equalityTest",2);
		assertTrue(
				"different picture returned for same subject (and index)",
				firstPicture.equals(secondPicture));
	}
	
	/**
	 * Test that requests for different subjects return different images.
	 */
	@Test
	public void nonequalityTest() {
		
		Picture firstPicture = proxy.getPicture("picture1",1);
		Picture secondPicture = proxy.getPicture("picture2",1);
		assertTrue(
				"same picture returned for different subject",
				!firstPicture.equals(secondPicture));
	}
	
	/**
	 * Test that the picture returned from the cache proxy has the
	 * index that was requested.
	 */
	@Test
	public void indexTest() {

		Picture firstPicture = proxy.getPicture("indexTest",100);
		Picture secondPicture = proxy.getPicture("indexTest",1234);
		assertTrue(
				"requested index 100 but retrieved index " + firstPicture.index(),
				firstPicture.index().equals(100));
		assertTrue(
				"requested index 1234 but retrieved index " + secondPicture.index(),
				secondPicture.index().equals(1234));
	}
	
	/**
	 * Test that the picture returned from the cache proxy matches the
	 * subject that was requested.
	 */
	@Test
	public void subjectTest() {

		Picture firstPicture = proxy.getPicture("subject1",2);
		Picture secondPicture = proxy.getPicture("subject2",2);
		assertTrue(
				"requested subject 'subject1' but retrieved subject " + firstPicture.subject(),
				firstPicture.subject().equals("subject1"));
		assertTrue(
				"requested subject 'subject2' but retrieved subject " + secondPicture.subject(),
				secondPicture.subject().equals("subject2"));
	}

	/**
	 * Return a picture from the simulated service.
	 * This service simply returns an empty picture every time that it called.
	 * Note that a <em>different</em> object is returned each time, even if the
	 * subject and index are the same.
	 *
	 * @param subject the requested subject
	 * @param index the index of the picture within all pictures for the requested topic
	 *
	 * @return the picture
	 */
	@Override
	public Picture getPicture(String subject, int index) {
		return new Picture((BufferedImage)null, subject ,serviceName(), index);
	}
	
	/**
	 * Return a textual name to identify the simulated service.
	 *
	 * @return the name of the service ("cacheProxyTest")
	 */
	@Override
	public String serviceName() {
		return "MyCacheProxyTest";
	}
}
