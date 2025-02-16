"use client"

import { useState, useRef, useCallback } from "react"
import Webcam from "react-webcam"
import { Button } from "@/components/ui/button"
import { MapIcon, Camera, RefreshCw } from "lucide-react"

const mirrorClass = "scale-x-[-1]"

interface ClassificationResult {
  trash_type: string
  disposal_bin: string
}

export default function TrashClassification() {
  const webcamRef = useRef<Webcam>(null)
  const [capturedImage, setCapturedImage] = useState<string | null>(null)
  const [isCapturing, setIsCapturing] = useState(true)
  const [classificationResult, setClassificationResult] = useState<ClassificationResult | null>(null)
  const [isClassifying, setIsClassifying] = useState(false)

  const capture = useCallback(() => {
    if (isCapturing) {
      const imageSrc = webcamRef.current?.getScreenshot()
      if (imageSrc) {
        // Create a new image to flip horizontally
        const img = new Image()
        img.onload = () => {
          const canvas = document.createElement("canvas")
          canvas.width = img.width
          canvas.height = img.height
          const ctx = canvas.getContext("2d")
          if (ctx) {
            ctx.scale(-1, 1) // Flip horizontally
            ctx.drawImage(img, -img.width, 0)
            const mirroredImageSrc = canvas.toDataURL()
            setCapturedImage(mirroredImageSrc)
            setIsCapturing(false)
            classifyImage(mirroredImageSrc)
          }
        }
        img.src = imageSrc
      }
    } else {
      setCapturedImage(null)
      setClassificationResult(null)
      setIsCapturing(true)
    }
  }, [isCapturing])

  const classifyImage = async (imageSrc: string) => {
    setIsClassifying(true)
    try {
      // Convert base64 to blob
      const response = await fetch(imageSrc)
      const blob = await response.blob()

      // Create FormData and append the file
      const formData = new FormData()
      formData.append("file", blob, "image.jpg")

      // Send POST request
      const classificationResponse = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData,
      })

      if (!classificationResponse.ok) {
        throw new Error("Classification request failed")
      }

      const result: ClassificationResult = await classificationResponse.json()
      setClassificationResult(result)
    } catch (error) {
      console.error("Error classifying image:", error)
      setClassificationResult(null)
    } finally {
      setIsClassifying(false)
    }
  }

  const getBinImage = (disposalBin: string) => {
    switch (disposalBin.toLowerCase()) {
      case "recycling":
        return "/bluebin.jpg"
      case "organic":
        return "/greenbin.jpg"
      case "landfill":
        return "/graybin.jpg"
      default:
        return "/placeholder.svg"
    }
  }

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">TheThreeR's Trash Classification System</h1>
      <div className="flex flex-col md:flex-row gap-4">
        <div className="flex-1">
          <div className="relative aspect-video">
            {isCapturing ? (
              <Webcam
                audio={false}
                ref={webcamRef}
                screenshotFormat="image/jpeg"
                className={`rounded-lg w-full h-full object-cover ${mirrorClass}`}
                mirrored={true}
              />
            ) : (
              <img
                src={capturedImage || "/placeholder.svg"}
                alt="Captured"
                className={`rounded-lg w-full h-full object-cover ${mirrorClass}`}
              />
            )}
          </div>
          <Button onClick={capture} className="mt-2 w-full" disabled={isClassifying}>
            {isCapturing ? (
              <>
                <Camera className="mr-2 h-4 w-4" /> Capture Image
              </>
            ) : (
              <>
                <RefreshCw className="mr-2 h-4 w-4" /> Take Another Picture
              </>
            )}
          </Button>
          <p className="mt-2 text-center text-sm text-gray-600">
            {isCapturing
              ? "Live webcam feed. Click the button to capture and classify trash."
              : "Image captured. Click the button to take another picture."}
          </p>
        </div>
        <div className="flex-1">
          <div className="bg-gray-200 rounded-lg aspect-video flex items-center justify-center overflow-hidden">
            {classificationResult ? (
              <img
                src={getBinImage(classificationResult.disposal_bin) || "/placeholder.svg"}
                alt={`${classificationResult.disposal_bin} bin`}
                className="object-cover w-full h-full"
              />
            ) : (
              <MapIcon className="h-16 w-16 text-gray-400" />
            )}
          </div>
          <p className="mt-2 text-center text-sm text-gray-600">
            {classificationResult
              ? `Disposal Bin: ${classificationResult.disposal_bin}`
              : "After you have captured an image, the system will identify the correct bin to dispose of your trash."}
          </p>
        </div>
      </div>
      {capturedImage && !isCapturing && (
        <div className="mt-4">
          <h2 className="text-xl font-semibold mb-2">Sorting Result</h2>
          {isClassifying ? (
            <p className="text-gray-600">Sorting trash...</p>
          ) : classificationResult ? (
            <div className="bg-white p-4 rounded-lg shadow">
              <p className="text-gray-800">
                <strong>Trash Type:</strong> {classificationResult.trash_type}
              </p>
              <p className="text-gray-800">
                <strong>Disposal Bin:</strong> {classificationResult.disposal_bin}
              </p>
            </div>
          ) : (
            <p className="text-gray-600">Failed to classify image. Please try again.</p>
          )}
        </div>
      )}
    </div>
  )
}

